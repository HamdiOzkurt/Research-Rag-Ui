'use client'

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function BillingRedirect() {
  const router = useRouter();
  useEffect(() => {
    router.replace("/?mode=billing");
  }, [router]);
  return null;
}


