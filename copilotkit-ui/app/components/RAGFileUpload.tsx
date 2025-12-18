'use client'

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Upload, FileText, Check, Loader2 } from "lucide-react";
import { uploadFile, ingestFile, listDocuments } from "@/lib/api";
import { Badge } from "@/components/ui/badge";

export default function RAGFileUpload() {
  const [isUploading, setIsUploading] = useState(false);
  const [documents, setDocuments] = useState<any[]>([]);

  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    try {
      const res = await listDocuments();
      if (res.documents) {
        setDocuments(res.documents);
      }
    } catch (e) {
      console.error("Failed to load documents", e);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Check for duplicate filename
    const isDuplicate = documents.some(doc => doc.filename === file.name);
    if (isDuplicate) {
      const confirmed = window.confirm(
        `⚠️ "${file.name}" zaten yüklenmiş!\n\nTekrar yüklemek istiyor musunuz? (Duplicate chunk'lar eklenebilir)`
      );
      if (!confirmed) {
        e.target.value = ""; // Reset input
        return;
      }
    }

    setIsUploading(true);
    try {
      // 1. Upload
      console.log("📤 Uploading file:", file.name);
      const uploadRes = await uploadFile(file);
      console.log("✅ Upload response:", uploadRes);
      
      if (uploadRes.status === "success") {
        // 2. Ingest
        console.log("🔄 Starting ingestion...", uploadRes.path);
        const ingestRes = await ingestFile(uploadRes.path, file.name);
        console.log("✅ Ingest response:", ingestRes);
        
        // Check if duplicate
        if (ingestRes.duplicate || ingestRes.status === "duplicate") {
          alert(`⚠️ Bu dosya zaten yüklenmişti!\n\n"${file.name}" daha önce ingest edilmiş.\n\nYeni chunk eklenmedi (0 chunk).`);
        } else if (ingestRes.chunks > 0) {
          console.log(`✅ ${ingestRes.chunks} chunks added`);
        }
        
        await loadDocuments();
        console.log("✅ Documents reloaded");
      } else {
        console.error("❌ Upload failed:", uploadRes);
        alert(`Upload failed: ${JSON.stringify(uploadRes)}`);
      }
    } catch (error) {
      console.error("❌ Error during upload/ingest:", error);
      alert(`Error: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsUploading(false);
      // Reset input
      e.target.value = "";
    }
  };

  return (
    <div className="p-4 border rounded-lg bg-muted/30 space-y-4 mb-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium flex items-center gap-2">
          <FileText className="w-4 h-4" />
          Dokümanlar (RAG Context)
        </h3>
        <div className="relative">
          <input
            type="file"
            id="rag-upload"
            className="hidden"
            onChange={handleFileChange}
            disabled={isUploading}
            accept=".pdf,.docx,.csv,.txt,.md,.mp3,.wav,.jpg,.png"
          />
          <Button
            variant="outline"
            size="sm"
            disabled={isUploading}
            onClick={() => document.getElementById("rag-upload")?.click()}
          >
            {isUploading ? (
              <Loader2 className="w-4 h-4 animate-spin mr-2" />
            ) : (
              <Upload className="w-4 h-4 mr-2" />
            )}
            {isUploading ? "İşleniyor..." : "Dosya Yükle"}
          </Button>
        </div>
      </div>

      {documents.length > 0 ? (
        <div className="flex flex-wrap gap-2">
          {documents.map((doc) => (
            <Badge key={doc.path} variant="secondary" className="flex items-center gap-1 pl-2 pr-1 py-1">
              <FileText className="w-3 h-3 opacity-70" />
              <span className="max-w-[150px] truncate" title={doc.filename}>{doc.filename}</span>
              <div className="bg-green-500/10 text-green-600 rounded-full p-0.5 ml-1">
                <Check className="w-3 h-3" />
              </div>
            </Badge>
          ))}
        </div>
      ) : (
        <div className="text-xs text-muted-foreground text-center py-2">
          Henüz doküman yüklenmedi. PDF, Word, Ses veya Görsel yükleyebilirsiniz.
        </div>
      )}
    </div>
  );
}
