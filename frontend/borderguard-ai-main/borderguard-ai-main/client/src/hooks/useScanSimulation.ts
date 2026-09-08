/*
 * BorderGuard AI scan state machine.
 * Driven by the real backend API call — not a fake timer.
 * Each step is advanced explicitly via advance() or completed via complete().
 */
import { useCallback, useMemo, useState } from "react";

export type ScanState =
  | "IDLE"
  | "DOCUMENT_DETECTED"
  | "OCR_EXTRACTION"
  | "DOCUMENT_VALIDATION"
  | "TAMPERING_ANALYSIS"
  | "FACE_VERIFICATION"
  | "RISK_ANALYSIS"
  | "VERIFIED"
  | "FAILED";

const sequence: ScanState[] = [
  "DOCUMENT_DETECTED",
  "OCR_EXTRACTION",
  "DOCUMENT_VALIDATION",
  "TAMPERING_ANALYSIS",
  "FACE_VERIFICATION",
  "RISK_ANALYSIS",
  "VERIFIED",
];

const labels: Record<ScanState, string> = {
  IDLE: "Ready for a document",
  DOCUMENT_DETECTED: "Document detected",
  OCR_EXTRACTION: "OCR extraction",
  DOCUMENT_VALIDATION: "Document validation",
  TAMPERING_ANALYSIS: "Tampering analysis",
  FACE_VERIFICATION: "Face verification",
  RISK_ANALYSIS: "Risk analysis",
  VERIFIED: "Identity verified",
  FAILED: "Scan failed",
};

export function useScanSimulation() {
  const [scanState, setScanState] = useState<ScanState>("IDLE");
  const [progress, setProgress] = useState(0);

  const startScan = useCallback(() => {
    setScanState("DOCUMENT_DETECTED");
    setProgress(6);
  }, []);

  const advance = useCallback(() => {
    setScanState((current) => {
      if (current === "IDLE" || current === "VERIFIED" || current === "FAILED") return current;
      const currentIndex = sequence.indexOf(current);
      if (currentIndex < 0 || currentIndex >= sequence.length - 1) return current;
      const next = sequence[currentIndex + 1];
      setProgress(Math.min(100, (currentIndex + 2) * 14));
      return next;
    });
  }, []);

  const complete = useCallback(() => {
    setScanState("VERIFIED");
    setProgress(100);
  }, []);

  const fail = useCallback(() => {
    setScanState("FAILED");
  }, []);

  const resetScan = useCallback(() => {
    setScanState("IDLE");
    setProgress(0);
  }, []);

  return useMemo(
    () => ({
      scanState,
      progress: scanState === "VERIFIED" ? 100 : progress,
      label: labels[scanState],
      isRunning: scanState !== "IDLE" && scanState !== "VERIFIED" && scanState !== "FAILED",
      isVerified: scanState === "VERIFIED",
      startScan,
      advance,
      complete,
      fail,
      resetScan,
    }),
    [progress, resetScan, scanState, startScan, advance, complete, fail],
  );
}
