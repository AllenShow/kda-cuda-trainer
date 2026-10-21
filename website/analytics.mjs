const HOST = "kda-cuda-trainer.pages.dev";
const DAY_MS = 86_400_000;
const CACHE_SECONDS = 300;
const REGION_LIMIT = 300;
const NUMBER = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const REGION_NAMES = new Intl.DisplayNames(["en"], { type: "region" });

// Cloudflare uses the lowercase `string` scalar. Keep host and site filters
// on both selections so other websites in the account cannot enter the card.
export const QUERY = `query WebsiteStats($accountTag: string!, $filter: AccountRumPageloadEventsAdaptiveGroupsFilter_InputObject!) {
  viewer {
    accounts(filter: { accountTag: $accountTag }) {
      total: rumPageloadEventsAdaptiveGroups(limit: 1, filter: $filter) {
        count
        sum { visits }
      }
      countries: rumPageloadEventsAdaptiveGroups(
        limit: ${REGION_LIMIT}, filter: $filter, orderBy: [count_DESC]
      ) {
        count
        dimensions { countryName }
      }
    }
  }
}`;

function configuration(env) {
  const account = env.CF_ACCOUNT_ID?.trim();
  const site = env.CF_WEB_ANALYTICS_SITE_TAG?.trim();
  const token = env.CF_ANALYTICS_API_TOKEN?.trim();
  if (!/^[a-f\d]{32}$/i.test(account ?? "") ||
      !/^[a-f\d]{32}$/i.test(site ?? "") || !token) return null;
  return { account, site, token };
}

function metric(value) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new Error("invalid_metric");
  }
  return value;
}

export function parseAnalytics(payload) {
  if (payload.errors?.length) throw new Error("graphql_error");
  const accounts = payload.data?.viewer?.accounts;
  if (!Array.isArray(accounts) || accounts.length !== 1) throw new Error("invalid_account");
  const { total, countries } = accounts[0];
  if (!Array.isArray(total) || total.length > 1 || !Array.isArray(countries) ||
      countries.length >= REGION_LIMIT || (!total.length && countries.length)) {
    throw new Error("invalid_groups");
  }
  const pageViews = total.length ? metric(total[0].count) : 0;
  const visits = total.length ? metric(total[0].sum?.visits) : 0;
  const regions = countries.map((row) => ({
    code: typeof row.dimensions?.countryName === "string" ? row.dimensions.countryName : "",
    pageViews: metric(row.count),
  })).filter((row) => row.pageViews > 0).sort((a, b) => b.pageViews - a.pageViews);
  return { pageViews, visits, regions };
}

function escapeXml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;",
  })[char]).replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F]/g, "");
}

function regionName(code) {
  if (!code || ["XX", "T1", "ZZ"].includes(code.toUpperCase())) return "Unknown region";
  if (/^[a-z]{2}$/i.test(code)) return REGION_NAMES.of(code.toUpperCase()) ?? code;
  return code.slice(0, 32);
}

function svgDocument(content, height, description) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="720" height="${height}" viewBox="0 0 720 ${height}" role="img" aria-labelledby="title desc">
  <title id="title">KDA CUDA Trainer website traffic</title>
  <desc id="desc">${escapeXml(description)}</desc>
  <style>
    text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #172033; }
    .muted { fill: #526078; font-size: 13px; }
    .label { fill: #526078; font-size: 14px; }
    .value { font-size: 30px; font-weight: 650; }
    .region { font-size: 14px; }
  </style>
  <rect x="1" y="1" width="718" height="${height - 2}" rx="14" fill="#f8fafc" stroke="#d8e0ec"/>
  <text x="28" y="36" font-size="19" font-weight="650">Website traffic · Last 30 days</text>
  ${content}
</svg>`;
}

export function renderUnavailable() {
  return svgDocument(`<text x="28" y="76" class="label">Statistics are currently unavailable.</text>
    <text x="28" y="104" class="muted">Visit the project website again later for traffic and regional statistics.</text>`,
  132, "Website statistics are currently unavailable. No traffic figures are being reported.");
}

export function renderStats(stats, start, end) {
  const topRegions = stats.regions.slice(0, 10);
  const height = 300 + Math.max(topRegions.length, 1) * 30;
  const regionRows = topRegions.map((region, index) => {
    const y = 222 + index * 30;
    const share = stats.pageViews > 0 ? Math.min(100, region.pageViews / stats.pageViews * 100) : 0;
    const name = regionName(region.code);
    const label = name.length > 30 ? `${name.slice(0, 29)}…` : name;
    return `<text x="28" y="${y}" class="region"><title>${escapeXml(name)}</title>${escapeXml(label)}</text>
      <rect x="288" y="${y - 11}" width="220" height="12" rx="6" fill="#e5eaf2"/>
      <rect x="288" y="${y - 11}" width="${(220 * share / 100).toFixed(1)}" height="12" rx="6" fill="#5474df"/>
      <text x="603" y="${y}" text-anchor="end" class="region">${NUMBER.format(region.pageViews)}</text>
      <text x="692" y="${y}" text-anchor="end" class="region">${share.toFixed(1)}%</text>`;
  }).join("\n");
  const range = `${start.toISOString().slice(0, 10)} to ${end.toISOString().slice(0, 10)} UTC`;
  return svgDocument(`
    <text x="28" y="60" class="muted">${range} · ${HOST}</text>
    <text x="28" y="96" class="label">Page views (PV)</text>
    <text x="28" y="134" class="value">${NUMBER.format(stats.pageViews)}</text>
    <text x="274" y="96" class="label">Visits</text>
    <text x="274" y="134" class="value">${NUMBER.format(stats.visits)}</text>
    <text x="520" y="96" class="label">Countries / regions</text>
    <text x="520" y="134" class="value">${stats.regions.filter((r) => r.code && !["XX", "T1", "ZZ"].includes(r.code.toUpperCase())).length}</text>
    <path d="M28 157H692" stroke="#d8e0ec"/>
    <text x="28" y="188" class="label">Top countries / regions</text>
    <text x="603" y="188" text-anchor="end" class="label">Page views</text>
    <text x="692" y="188" text-anchor="end" class="label">Share of PV</text>
    ${regionRows || '<text x="28" y="222" class="muted">No page views recorded in this period.</text>'}
    <text x="28" y="${height - 54}" class="muted">Cloudflare Web Analytics · Counts may be estimated · Refreshed every 5 minutes</text>
    <text x="28" y="${height - 32}" class="muted">Visits are not unique visitors (UV). GitHub repository views are excluded.</text>
    <text x="28" y="${height - 10}" class="muted">Updated ${escapeXml(end.toISOString().replace("T", " ").replace(".000Z", " UTC"))}</text>`,
  height, `${NUMBER.format(stats.pageViews)} page views, ${NUMBER.format(stats.visits)} visits. Country and region shares are based on all website page views in the last 30 days. ${range}.`);
}

function svgResponse(svg, available) {
  return new Response(svg, { headers: {
    "Content-Type": "image/svg+xml; charset=utf-8",
    "Cache-Control": available ? `public, max-age=${CACHE_SECONDS}` : "no-store",
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; sandbox",
  } });
}

export async function handleStats(context, { fetcher = fetch, cache = globalThis.caches?.default, now = () => new Date() } = {}) {
  const { request, env } = context;
  if (!["GET", "HEAD"].includes(request.method)) {
    return new Response("Method not allowed", { status: 405, headers: { Allow: "GET, HEAD" } });
  }
  const config = configuration(env);
  if (!config) {
    const response = svgResponse(renderUnavailable(), false);
    return request.method === "HEAD" ? new Response(null, response) : response;
  }
  // Ignore arbitrary query strings so image cache-busters cannot multiply API calls.
  const key = new Request(new URL("/stats.svg", request.url), { method: "GET" });
  let response;
  try {
    response = await cache?.match(key);
  } catch {
    // A cache outage must not prevent fetching valid analytics.
  }
  if (!response) {
    try {
      // A five-minute boundary gives a stable time window to both homepages.
      const end = new Date(Math.floor(now().getTime() / (CACHE_SECONDS * 1000)) * CACHE_SECONDS * 1000);
      const start = new Date(end.getTime() - 30 * DAY_MS);
      const upstream = await fetcher("https://api.cloudflare.com/client/v4/graphql", {
        method: "POST",
        headers: { "Authorization": `Bearer ${config.token}`, "Content-Type": "application/json" },
        body: JSON.stringify({ query: QUERY, variables: {
          accountTag: config.account,
          filter: {
            siteTag: config.site, requestHost: HOST,
            datetime_geq: start.toISOString(), datetime_lt: end.toISOString(),
          },
        } }),
        signal: AbortSignal.timeout(10_000),
      });
      if (!upstream.ok) throw new Error("upstream_http_error");
      const stats = parseAnalytics(await upstream.json());
      response = svgResponse(renderStats(stats, start, end), true);
      if (cache) {
        const put = cache.put(key, response.clone()).catch(() => {});
        if (context.waitUntil) context.waitUntil(put);
        else await put;
      }
    } catch {
      // Never return API error bodies, account identifiers, or credentials publicly.
      console.error("Website analytics could not be loaded; check Pages analytics bindings and API access.");
      response = svgResponse(renderUnavailable(), false);
    }
  }
  return request.method === "HEAD" ? new Response(null, response) : response;
}
