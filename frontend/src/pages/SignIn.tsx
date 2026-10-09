import { KeyRound } from "lucide-react";
import { useState } from "react";

import { signIn } from "@/api/client";
import { Button } from "@/components/ui/button";
import { Card, CardBody } from "@/components/ui/card";
import { Field, Input } from "@/components/ui/field";

export function SignIn({ onSignedIn }: { onSignedIn: () => void }) {
  const [token, setToken] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <Card className="w-full max-w-md">
        <CardBody className="space-y-4 py-6">
          <div className="flex items-center gap-3">
            <span className="flex size-10 items-center justify-center rounded-full bg-accent-soft text-accent">
              <KeyRound className="size-5" />
            </span>
            <div>
              <h1 className="text-lg font-semibold">Sign in to Research Tool</h1>
              <p className="text-sm text-ink-soft">This server only opens for the person who started it.</p>
            </div>
          </div>
          <p className="text-sm">
            Open the link that <code className="rounded bg-muted px-1">rtool ui</code> printed (it ends in{" "}
            <code className="rounded bg-muted px-1">#token=…</code>), or paste the token here.
          </p>
          <form
            className="space-y-3"
            onSubmit={async (event) => {
              event.preventDefault();
              setBusy(true);
              setError("");
              try {
                await signIn(token.trim());
                onSignedIn();
              } catch {
                setError("That token was not accepted. Check that the server is still running.");
              } finally {
                setBusy(false);
              }
            }}
          >
            <Field label="Access token" htmlFor="token">
              <Input id="token" value={token} onChange={(e) => setToken(e.target.value)} autoComplete="off" autoFocus />
            </Field>
            {error && (
              <p role="alert" className="text-sm text-bad">
                {error}
              </p>
            )}
            <Button type="submit" className="w-full" busy={busy} disabled={!token.trim()}>
              Sign in
            </Button>
          </form>
        </CardBody>
      </Card>
    </div>
  );
}
