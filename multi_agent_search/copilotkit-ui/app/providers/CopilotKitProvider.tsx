'use client'

import { CopilotKit } from "@copilotkit/react-core";
import "@copilotkit/react-ui/styles.css";

export function CopilotKitProvider({ children }: { children: React.ReactNode }) {
  return (
    <CopilotKit
      runtimeUrl={process.env.NEXT_PUBLIC_COPILOTKIT_RUNTIME_URL || "http://127.0.0.1:8000/copilotkit"}
      publicApiKey={process.env.NEXT_PUBLIC_COPILOTKIT_PUBLIC_KEY}
      showDevConsole={process.env.NODE_ENV === 'development'}
    >
      {children}
    </CopilotKit>
  );
}

