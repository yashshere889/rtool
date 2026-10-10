import { Link } from "react-router";

import { Empty } from "@/components/ui/states";

export function NotFound() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-10">
      <Empty
        title="Page not found"
        action={
          <Link to="/" className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent">
            Go to your projects
          </Link>
        }
      />
    </div>
  );
}
