export function retryDelay(attempt, baseMs, maxMs) {
  return Math.min(maxMs, baseMs * (2 ** (attempt + 1)));
}
