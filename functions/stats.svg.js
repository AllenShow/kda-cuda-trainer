import { handleStats } from "../website/analytics.mjs";

export function onRequest(context) {
  return handleStats(context);
}
