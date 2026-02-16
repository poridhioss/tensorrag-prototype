"use client";

import type { JsonSchema, JsonSchemaProperty } from "@/lib/types";

interface SchemaFormProps {
  schema: JsonSchema;
  values: Record<string, unknown>;
  onChange: (values: Record<string, unknown>) => void;
}

function renderField(
  key: string,
  prop: JsonSchemaProperty,
  value: unknown,
  update: (key: string, val: unknown) => void
) {
  // String with enum -> select
  if (prop.type === "string" && prop.enum) {
    return (
      <select
        value={(value as string) ?? ""}
        onChange={(e) => update(key, e.target.value)}
        className="w-full rounded-md border border-border bg-background px-2 py-1.5 text-xs"
      >
        <option value="">Select...</option>
        {prop.enum.map((opt) => (
          <option key={String(opt)} value={String(opt)}>
            {String(opt)}
          </option>
        ))}
      </select>
    );
  }

  // Boolean -> checkbox
  if (prop.type === "boolean") {
    return (
      <label className="flex items-center gap-2 text-xs">
        <input
          type="checkbox"
          checked={!!value}
          onChange={(e) => update(key, e.target.checked)}
          className="rounded border-border"
        />
        <span>{prop.description || key}</span>
      </label>
    );
  }

  // Number
  if (prop.type === "number" || prop.type === "integer") {
    return (
      <input
        type="number"
        step={prop.type === "integer" ? 1 : "any"}
        value={value !== undefined && value !== null ? String(value) : ""}
        onChange={(e) => {
          const v = e.target.value;
          update(key, v === "" ? undefined : Number(v));
        }}
        className="w-full rounded-md border border-border bg-background px-2 py-1.5 text-xs"
      />
    );
  }

  // Array of strings with enum -> multi-select checkboxes
  if (
    prop.type === "array" &&
    prop.items?.type === "string" &&
    prop.items?.enum
  ) {
    const selected = Array.isArray(value) ? (value as string[]) : [];
    return (
      <div className="flex flex-wrap gap-1.5">
        {prop.items.enum.map((opt) => (
          <label key={opt} className="flex items-center gap-1 text-xs">
            <input
              type="checkbox"
              checked={selected.includes(opt)}
              onChange={(e) => {
                const next = e.target.checked
                  ? [...selected, opt]
                  : selected.filter((s) => s !== opt);
                update(key, next);
              }}
              className="rounded border-border"
            />
            {opt}
          </label>
        ))}
      </div>
    );
  }

  // Array of strings without enum -> comma-separated input
  if (prop.type === "array" && prop.items?.type === "string") {
    const arr = Array.isArray(value) ? (value as string[]) : [];
    return (
      <input
        type="text"
        value={arr.join(", ")}
        onChange={(e) => {
          const parts = e.target.value
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean);
          update(key, parts.length > 0 ? parts : []);
        }}
        placeholder="comma-separated values"
        className="w-full rounded-md border border-border bg-background px-2 py-1.5 text-xs"
      />
    );
  }

  // Object -> JSON textarea
  if (prop.type === "object") {
    const strVal =
      typeof value === "string" ? value : JSON.stringify(value || {}, null, 2);
    return (
      <textarea
        value={strVal}
        onChange={(e) => {
          try {
            update(key, JSON.parse(e.target.value));
          } catch {
            // Keep as string while editing
          }
        }}
        rows={3}
        className="w-full rounded-md border border-border bg-background px-2 py-1.5 text-xs font-mono"
        placeholder="{}"
      />
    );
  }

  // Default: text input
  return (
    <input
      type="text"
      value={(value as string) ?? ""}
      onChange={(e) => update(key, e.target.value || undefined)}
      className="w-full rounded-md border border-border bg-background px-2 py-1.5 text-xs"
    />
  );
}

export function SchemaForm({ schema, values, onChange }: SchemaFormProps) {
  // Support both JSON Schema format (with "properties" wrapper) and
  // flat format where keys are directly on the schema object
  const properties: Record<string, JsonSchemaProperty> | undefined =
    schema.properties ??
    (Object.keys(schema).some(
      (k) =>
        !["type", "required", "description", "title", "default"].includes(k)
    )
      ? (schema as unknown as Record<string, JsonSchemaProperty>)
      : undefined);

  if (!properties) return null;

  const updateField = (key: string, val: unknown) => {
    onChange({ ...values, [key]: val });
  };

  // Filter out JSON Schema meta-keys when using flat format
  const schemaMetaKeys = new Set(["type", "required", "description", "title", "default"]);

  return (
    <div className="space-y-3">
      {Object.entries(properties)
        .filter(([key]) => !key.startsWith("_") && !schemaMetaKeys.has(key))
        .filter(([, prop]) => prop && typeof prop === "object" && "type" in prop)
        .map(([key, prop]) => (
          <div key={key}>
            {prop.type !== "boolean" && (
              <label className="block text-xs font-medium mb-1">
                {key.replace(/_/g, " ")}
                {schema.required?.includes(key) && (
                  <span className="text-status-failed ml-0.5">*</span>
                )}
              </label>
            )}
            {renderField(key, prop, values[key], updateField)}
            {prop.description && prop.type !== "boolean" && (
              <p className="text-[10px] text-text-secondary mt-0.5">
                {prop.description}
              </p>
            )}
          </div>
        ))}
    </div>
  );
}
