'use client'

import { CopilotKit } from "@copilotkit/react-core";
import "@copilotkit/react-ui/styles.css";

export function CopilotKitProvider({ children }: { children: React.ReactNode }) {
  return (
    <CopilotKit
      publicApiKey={process.env.NEXT_PUBLIC_COPILOTKIT_PUBLIC_KEY}
      // Eğer key yoksa, cloud runtime kullanmadan çalışsın
      showDevConsole={process.env.NODE_ENV === 'development'}
    >
      {children}
    </CopilotKit>
  );
}
