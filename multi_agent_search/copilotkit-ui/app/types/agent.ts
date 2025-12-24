export type AgentMode = "simple" | "multi" | "deep";
export interface AgentStatus {
    status: "initializing" | "planning" | "searching" | "researching" | "coding" | "writing" | "done" | "error";
    message: string;
    thread_id?: string;
    run_id?: string;
    meta?: any;
}
