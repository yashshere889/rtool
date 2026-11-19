import { useEffect } from "react";

const APP = "Research Tool";

/** Sets the browser tab's title while the page is shown (``null``: just the app name). */
export function useTitle(title: string | null) {
  useEffect(() => {
    document.title = title ? `${title} · ${APP}` : APP;
    return () => {
      document.title = APP;
    };
  }, [title]);
}
