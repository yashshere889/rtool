import { CheckCircle2, XCircle } from "lucide-react";
import { useSearchParams } from "react-router";

import { useSettings } from "@/api/hooks";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { ErrorNote, Loading } from "@/components/ui/states";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import { ChecksTab, FieldsTab, PipelinesTab, ValidityTab } from "./settings/SystemTabs";

const KEY_HELP: Record<string, string> = {
  OPENALEX_API_KEY: "OpenAlex search (free key from openalex.org)",
  S2_API_KEY: "Semantic Scholar (free key; higher rate limit)",
  UNPAYWALL_EMAIL: "Unpaywall lookups for open-access copies",
  HF_TOKEN: "Only for gated Hugging Face datasets and models",
};

export function Settings() {
  const settings = useSettings();
  const [params, setParams] = useSearchParams();
  if (settings.isPending) return <Loading />;
  if (settings.isError) return <ErrorNote error={settings.error} />;
  const s = settings.data;
  return (
    <div className="mx-auto max-w-4xl space-y-6 px-4 py-8">
      <h1 className="text-2xl font-semibold">Settings</h1>
      <Tabs value={params.get("tab") ?? "server"} onValueChange={(tab) => setParams(tab === "server" ? {} : { tab })}>
        <TabsList>
          <TabsTrigger value="server">This server</TabsTrigger>
          <TabsTrigger value="fields">Fields</TabsTrigger>
          <TabsTrigger value="pipelines">Pipelines & agents</TabsTrigger>
          <TabsTrigger value="validity">Validity</TabsTrigger>
          <TabsTrigger value="checks">Checks</TabsTrigger>
        </TabsList>
        <TabsContent value="server" className="mt-4 space-y-6">
      <Card>
        <CardHeader title="This server" />
        <CardBody className="space-y-2 text-sm">
          <p>
            <span className="text-ink-soft">Mode:</span>{" "}
            {s.runs_locally ? "local demo (papers, model and cluster simulated)" : `on the cluster (${s.mode})`}
          </p>
          <p className="break-all">
            <span className="text-ink-soft">Runs folder:</span> {s.runs_root}
          </p>
          <p>
            <span className="text-ink-soft">Version:</span> {s.version}
          </p>
        </CardBody>
      </Card>
      <Card>
        <CardHeader
          title="API keys"
          description="Read from ~/.config/rtool/secrets.env on the server. Only whether each is set is shown, never its value."
        />
        <CardBody>
          <ul className="space-y-2 text-sm">
            {Object.entries(s.secrets).map(([name, present]) => (
              <li key={name} className="flex items-start gap-2">
                {present ? <CheckCircle2 className="mt-0.5 size-4 text-ok" /> : <XCircle className="mt-0.5 size-4 text-ink-soft" />}
                <span>
                  <code className="font-mono">{name}</code> <span className="text-ink-soft">— {KEY_HELP[name] ?? ""}</span>
                </span>
              </li>
            ))}
          </ul>
        </CardBody>
      </Card>
        </TabsContent>
        <TabsContent value="fields" className="mt-4">
          <FieldsTab />
        </TabsContent>
        <TabsContent value="pipelines" className="mt-4">
          <PipelinesTab />
        </TabsContent>
        <TabsContent value="validity" className="mt-4">
          <ValidityTab />
        </TabsContent>
        <TabsContent value="checks" className="mt-4">
          <ChecksTab />
        </TabsContent>
      </Tabs>
    </div>
  );
}
