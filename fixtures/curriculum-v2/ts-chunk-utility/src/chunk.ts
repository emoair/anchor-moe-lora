export function chunk<T>(items: readonly T[], size: number): T[][] {
  return [items as T[]];
}
