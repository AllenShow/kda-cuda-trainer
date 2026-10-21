# Website analytics

The project website and the root GitHub README embed the same public card:

`https://kda-cuda-trainer.pages.dev/stats.svg`

The card reports the preceding 30 days of website page views (PV), visits, and the
top ten countries / regions by page views, including each region's share of **all**
website page views. Cloudflare Visits are not deduplicated visitors (UV).
The card includes the UTC time range and update time. Cloudflare may sample data,
so counts can be estimates. These figures do not measure GitHub repository views.

## Configure the existing Cloudflare Pages project

1. Open **Workers & Pages → kda-cuda-trainer → Metrics**, and enable
   **Web Analytics** if it is not already enabled. Pages injects the tracking
   script on the next deployment; do not add a second beacon to the HTML.
2. Create a Cloudflare API token with **Account → Account Analytics → Read**, scoped
   to the account that owns this Pages project. A read-only analytics token is
   sufficient; the deployed function does not need deployment permissions.
3. In **Pages → kda-cuda-trainer → Settings → Variables and Secrets**, select
   **Production** and add the following bindings:

   | Binding | Value | Storage |
   | --- | --- | --- |
   | `CF_ACCOUNT_ID` | The 32-character Cloudflare account ID | Text or secret |
   | `CF_WEB_ANALYTICS_SITE_TAG` | The 32-character **site tag** for this website's Web Analytics property | Text or secret |
   | `CF_ANALYTICS_API_TOKEN` | The account-scoped analytics read token from step 2 | **Encrypted secret** |

   The Web Analytics **site tag** is different from the public beacon token.
   It is the `siteTag` for the property selected in the Web Analytics dashboard.
   It can also be obtained from the `site_tag` field of Cloudflare's
   [List RUM sites API](https://developers.cloudflare.com/api/resources/rum/subresources/site_info/methods/list/)
   using a separately authorized credential. Do not substitute the token from
   the injected beacon script for this site tag.
4. Deploy the updated repository with the repository root as the Pages root
   directory and `docs/site` as the output directory. For the existing buildless
   site, leave the build command empty. The root `functions/` directory must be
   present at build time: uploading only `docs/site` from the dashboard does not
   build Pages Functions. The existing `wrangler.toml` declares the output path.
5. Redeploy **after** adding the bindings. If using previews, configure their
   bindings separately; bindings are intentionally not stored in Git.
6. Open `/stats.svg`. It should show numeric totals, a UTC update time, and regional
   rows (or a genuine no-page-views message when the successful query is empty).
   The website homepage and root README use this same URL.

中文：在现有 Pages 项目的 **设置 → 变量和机密 → Production** 中配置上面三个值，
将 `CF_ANALYTICS_API_TOKEN` 保存为加密机密，然后重新部署。这里只需要统计读取权限。
网站与 README 共用 `/stats.svg`，无需创建新的 Worker 或域名。

## Failure states and caching

- Missing / invalid configuration, access errors, GraphQL errors, timeouts, and
  invalid responses show **Statistics are currently unavailable**. No fabricated
  traffic figures or API error details are exposed to visitors.
- A successful empty query shows **0 page views**, **0 visits**, and no country rows.
- The endpoint caches successful SVG responses for five minutes at the Cloudflare
  location handling the request. GitHub proxies images and can delay visible updates.
- Query strings do not create separate server-side cache entries. The function
  accepts only GET / HEAD, does not ingest visitor events, and does not disclose
  credentials or account identifiers.
- The query is restricted to both the configured site tag and the exact hostname
  `kda-cuda-trainer.pages.dev`; it does not include unrelated account traffic or
  preview subdomains. Country percentages use the aggregate total rather than
  the sum of the top ten countries. Unknown locations remain labeled as unknown.

If the card remains unavailable after configuration, check the account ID, site tag,
token scope, and Pages Function logs. Confirm Web Analytics is collecting traffic.
The public error card deliberately omits upstream error bodies and secrets.

## Local checks

The server code has no npm dependencies. With Node.js 22 or later:

```sh
node --test tests/site/*.test.mjs
```

For local runtime validation, install Wrangler and run `wrangler pages dev docs/site`
from the repository root. Put the three bindings in an ignored `.dev.vars` file.
Never put an API token in `index.html`, README, `wrangler.toml`, or a tracked file.

## References

- [Enable Pages Web Analytics](https://developers.cloudflare.com/pages/how-to/web-analytics/)
- [Cloudflare Visits and Page views](https://developers.cloudflare.com/web-analytics/data-metrics/high-level-metrics/)
- [Country dimension](https://developers.cloudflare.com/web-analytics/data-metrics/dimensions/)
- [Create an analytics API token](https://developers.cloudflare.com/analytics/graphql-api/getting-started/authentication/api-token-auth/)
- [Pages secrets](https://developers.cloudflare.com/pages/functions/bindings/#secrets)
- [Capture dashboard GraphQL queries](https://developers.cloudflare.com/analytics/graphql-api/tutorials/capture-graphql-queries-from-dashboard/)
