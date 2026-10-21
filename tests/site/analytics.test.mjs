import assert from "node:assert/strict";
import { test } from "node:test";
import { readFile } from "node:fs/promises";
import { handleStats, parseAnalytics, renderStats } from "../../website/analytics.mjs";

const env = {
  CF_ACCOUNT_ID: "a".repeat(32),
  CF_WEB_ANALYTICS_SITE_TAG: "b".repeat(32),
  CF_ANALYTICS_API_TOKEN: "test-secret-never-render",
};
const end = new Date("2026-10-21T12:30:00Z");
const context = (method = "GET", bindings = env, query = "") => ({
  env: bindings,
  request: new Request(`https://kda-cuda-trainer.pages.dev/stats.svg${query}`, { method }),
});
const data = (total = [{ count: 100, sum: { visits: 40 } }], countries = [
  { count: 60, dimensions: { countryName: "CN" } },
  { count: 20, dimensions: { countryName: "US" } },
  { count: 10, dimensions: { countryName: "XX" } },
]) => ({ data: { viewer: { accounts: [{ total, countries }] } } });

test("query uses the configured site, exact host, and a bounded 30-day window", async () => {
  const response = await handleStats(context(), {
    now: () => new Date("2026-10-21T12:33:44Z"),
    fetcher: async (url, options) => {
      assert.equal(url, "https://api.cloudflare.com/client/v4/graphql");
      assert.equal(options.method, "POST");
      assert.equal(options.headers.Authorization, `Bearer ${env.CF_ANALYTICS_API_TOKEN}`);
      const { query, variables } = JSON.parse(options.body);
      assert.match(query, /dimensions \{ countryName \}/);
      assert.equal(variables.accountTag, env.CF_ACCOUNT_ID);
      assert.deepEqual(variables.filter, {
        siteTag: env.CF_WEB_ANALYTICS_SITE_TAG,
        requestHost: "kda-cuda-trainer.pages.dev",
        datetime_geq: "2026-09-21T12:30:00.000Z",
        datetime_lt: "2026-10-21T12:30:00.000Z",
      });
      return Response.json(data());
    },
  });
  const svg = await response.text();
  assert.match(response.headers.get("content-type"), /image\/svg\+xml/);
  assert.match(svg, /China/);
  assert.match(svg, /United States/);
  assert.match(svg, /Unknown region/);
  assert.match(svg, /60\.0%/); // The full total is 100; listed regions sum to only 90.
  assert.doesNotMatch(svg, /66\.7%/);
  assert.match(svg, /Visits are not unique visitors/);
  assert.ok(!svg.includes(env.CF_ANALYTICS_API_TOKEN));
  assert.ok(!svg.includes(env.CF_ACCOUNT_ID));
});

test("no configuration never queries the API and never reports invented zeros", async () => {
  const response = await handleStats(context("GET", {}), {
    fetcher: () => assert.fail("API must not be called without bindings"),
  });
  assert.equal(response.headers.get("cache-control"), "no-store");
  const svg = await response.text();
  assert.match(svg, /Statistics are currently unavailable/);
  assert.doesNotMatch(svg, /class="value"/);
});

test("a successful empty result is distinguished from failed statistics", async () => {
  const response = await handleStats(context(), {
    now: () => end,
    fetcher: async () => Response.json(data([], [])),
  });
  const svg = await response.text();
  assert.match(svg, /class="value">0</);
  assert.match(svg, /No page views recorded/);
  assert.doesNotMatch(svg, /currently unavailable/);
});

test("HTTP, GraphQL, malformed JSON, and network failures hide upstream details", async () => {
  const failures = [
    async () => new Response("private upstream error", { status: 403 }),
    async () => Response.json({ errors: [{ message: env.CF_ANALYTICS_API_TOKEN }], ...data() }),
    async () => new Response("not JSON"),
    async () => { throw new Error(env.CF_ANALYTICS_API_TOKEN); },
    async () => Response.json({ data: { viewer: { accounts: [] } } }),
  ];
  for (const fetcher of failures) {
    const response = await handleStats(context(), { fetcher });
    const svg = await response.text();
    assert.match(svg, /currently unavailable/);
    assert.ok(!svg.includes(env.CF_ANALYTICS_API_TOKEN));
    assert.ok(!svg.includes("private upstream error"));
    assert.equal(response.headers.get("cache-control"), "no-store");
  }
});

test("success is cached and arbitrary query strings reuse the same cache key", async () => {
  const entries = new Map();
  const cache = {
    match: async (key) => entries.get(key.url)?.clone(),
    put: async (key, response) => entries.set(key.url, response),
  };
  let calls = 0;
  const options = { cache, now: () => end, fetcher: async () => { calls += 1; return Response.json(data()); } };
  const first = await handleStats(context("GET", env, "?bust=1"), options);
  const second = await handleStats(context("GET", env, "?bust=2"), options);
  assert.equal(calls, 1);
  assert.equal(await first.text(), await second.text());
  assert.equal(first.headers.get("cache-control"), "public, max-age=300");
});

test("HEAD has the same headers without a body and other methods cannot trigger queries", async () => {
  for (const bindings of [env, {}]) {
    const response = await handleStats(context("HEAD", bindings), { fetcher: async () => Response.json(data()) });
    assert.equal(await response.text(), "");
    assert.match(response.headers.get("content-type"), /image\/svg\+xml/);
  }
  const response = await handleStats(context("POST"), { fetcher: () => assert.fail("must not query") });
  assert.equal(response.status, 405);
  assert.equal(response.headers.get("allow"), "GET, HEAD");
});

test("cache failures do not hide healthy analytics", async () => {
  const response = await handleStats(context(), {
    cache: {
      match: async () => { throw new Error("cache offline"); },
      put: async () => { throw new Error("cache offline"); },
    },
    fetcher: async () => Response.json(data()),
  });
  assert.match(await response.text(), /class="value">100</);
});

test("malformed totals and truncated country queries do not masquerade as valid counts", () => {
  assert.throws(() => parseAnalytics(data([{ count: -1, sum: { visits: 2 } }])));
  assert.throws(() => parseAnalytics(data([{ count: 3, sum: {} }])));
  assert.throws(() => parseAnalytics(data([], [{ count: 1, dimensions: { countryName: "CN" } }])));
  assert.throws(() => parseAnalytics(data(undefined, Array.from({ length: 300 }, () => ({ count: 1 })))));
});

test("country labels are escaped and top ten shares use the total across all regions", () => {
  const regions = Array.from({ length: 12 }, (_, index) => ({ code: `Region ${index}`, pageViews: 10 }));
  regions[0].code = '<script>alert("x")</script>';
  const svg = renderStats({ pageViews: 120, visits: 60, regions }, new Date("2026-09-21T12:30:00Z"), end);
  assert.doesNotMatch(svg, /<script>/);
  assert.match(svg, /&lt;script&gt;/);
  assert.match(svg, /8\.3%/);
  assert.doesNotMatch(svg, /Region 10|Region 11/);
});

test("website statistics and GitHub README image counters have separate scopes", async () => {
  const root = new URL("../../", import.meta.url);
  const [readme, html, routes] = await Promise.all([
    readFile(new URL("README.md", root), "utf8"),
    readFile(new URL("docs/site/index.html", root), "utf8"),
    readFile(new URL("docs/site/_routes.json", root), "utf8"),
  ]);
  assert.ok(readme.includes("https://kda-cuda-trainer.pages.dev/stats.svg"));
  assert.ok(html.includes('src="/stats.svg"'));
  assert.ok(readme.includes("## Official Website Traffic — Cloudflare"));
  assert.ok(readme.includes("## GitHub Repository Traffic — VisitorBadge & FlagCounter"));
  assert.ok(readme.includes("api.visitorbadge.io/api/visitors?path=AllenShow.kda-cuda-trainer&"));
  assert.ok(readme.includes("s05.flagcounter.com/count/WGbp/"));
  assert.ok(!html.includes("api.visitorbadge.io"));
  assert.ok(!html.includes("s05.flagcounter.com"));
  assert.deepEqual(JSON.parse(routes).include, ["/stats.svg"]);
});
