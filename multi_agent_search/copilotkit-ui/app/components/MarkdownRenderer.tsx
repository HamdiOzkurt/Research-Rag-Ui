'use client'

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { Button } from "@/components/ui/button";
import { Check, Copy } from "lucide-react";

type Props = {
  content: string;
};

export default function MarkdownRenderer({ content }: Props) {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const copyCode = (code: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(code);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  return (
    <div className="prose prose-sm dark:prose-invert max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          img({ node, src, alt, ...props }: any) {
            return (
              <span className="block my-4">
                <img 
                  src={src} 
                  alt={alt || 'Image'} 
                  className="max-w-full h-auto rounded-lg border border-gray-300 dark:border-gray-700 cursor-pointer hover:opacity-90 transition-opacity"
                  onClick={() => window.open(src, '_blank')}
                  {...props}
                />
                {alt && <em className="text-sm text-gray-500 dark:text-gray-400 block mt-2">{alt}</em>}
              </span>
            );
          },
          code({ node, className, children, ...props }: any) {
            const match = /language-(\w+)/.exec(className || "");
            const codeString = String(children).replace(/\n$/, "");
            const inline = !match;
            
            if (!inline && match) {
              return (
                <div className="relative group">
                  <Button
                    size="icon"
                    variant="ghost"
                    className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={() => copyCode(codeString)}
                  >
                    {copiedCode === codeString ? (
                      <Check className="w-4 h-4 text-emerald-500" />
                    ) : (
                      <Copy className="w-4 h-4" />
                    )}
                  </Button>
                  <pre className={className}>
                    <code className={className} {...props}>
                      {children}
                    </code>
                  </pre>
                </div>
              );
            }
            
            return (
              <code className={className} {...props}>
                {children}
              </code>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
