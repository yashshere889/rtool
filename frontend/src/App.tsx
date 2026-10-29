import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { createBrowserRouter } from "react-router";
import { RouterProvider } from "react-router/dom";

import { api, ApiError, onSignedOut, signIn } from "./api/client";
import { AppShell } from "./components/layout/AppShell";
import { RunLayout } from "./components/layout/RunLayout";
import { Loading } from "./components/ui/states";
import { TooltipProvider } from "./components/ui/tooltip";
import { NewResearch } from "./pages/NewResearch";
import { NotFound } from "./pages/NotFound";
import { PaperReader } from "./pages/literature/PaperReader";
import { Overview } from "./pages/Overview";
import { Projects } from "./pages/Projects";
import { Settings } from "./pages/Settings";
import { SignIn } from "./pages/SignIn";
import { StagePage } from "./pages/StagePage";

export const routes = [
  {
    element: <AppShell />,
    children: [
      { index: true, element: <Projects /> },
      { path: "new", element: <NewResearch /> },
      { path: "settings", element: <Settings /> },
      {
        path: "runs/:project/:runId",
        element: <RunLayout />,
        children: [
          { index: true, element: <Overview /> },
          { path: ":stageId", element: <StagePage /> },
          { path: ":stageId/reader", element: <PaperReader /> },
        ],
      },
      { path: "*", element: <NotFound /> },
    ],
  },
];

const router = createBrowserRouter(routes);

export function makeQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: (failures, error) =>
          !(error instanceof ApiError && [401, 403, 404, 422].includes(error.status)) && failures < 2,
      },
    },
  });
}

const queryClient = makeQueryClient();

type Session = "checking" | "in" | "out";

/** Trades a #token=… from the start-up link for a session cookie, then checks the session works. */
async function startSession(): Promise<Session> {
  const token = new URLSearchParams(window.location.hash.slice(1)).get("token");
  if (token) {
    try {
      await signIn(token);
    } catch {
      // a stale link: fall through to the sign-in page
    }
    window.history.replaceState(null, "", window.location.pathname + window.location.search);
  }
  try {
    await api("/settings");
    return "in";
  } catch (error) {
    return error instanceof ApiError && error.status === 401 ? "out" : "in";
  }
}

export function App() {
  const [session, setSession] = useState<Session>("checking");
  useEffect(() => {
    const unsubscribe = onSignedOut(() => setSession("out"));
    void startSession().then(setSession);
    return unsubscribe;
  }, []);

  if (session === "checking") return <Loading label="Connecting" />;
  if (session === "out")
    return (
      <SignIn
        onSignedIn={() => {
          queryClient.clear();
          setSession("in");
        }}
      />
    );
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider delayDuration={300}>
        <RouterProvider router={router} />
      </TooltipProvider>
    </QueryClientProvider>
  );
}
