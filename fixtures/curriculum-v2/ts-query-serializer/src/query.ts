type QueryValue = string | number | boolean | null | undefined;

export function toQuery(input: Readonly<Record<string, QueryValue | readonly QueryValue[]>>): string {
  return "";
}
