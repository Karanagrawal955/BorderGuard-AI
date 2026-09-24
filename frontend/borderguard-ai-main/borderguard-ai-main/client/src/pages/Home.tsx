/*
 * BorderGuard AI visual system: cinematic intelligence platform inspired by the supplied reference.
 * This page orchestrates the R3F scene, scan state machine, and LIVE verification UI with
 * real backend integration for image authenticity screening.
 * FIXED: Frontend → Upload → Verify/Scan → Existing Backend (Flask /api/analyze-document) → Real result → Existing UI
 * No terminal/CMD input required.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { flushSync } from "react-dom";
import {
  Activity, AlertTriangle, BarChart3, Bell, BookOpenCheck, Boxes, Check, ChevronRight, CircleDot,
  Clock3, Database, FileCheck2, FileScan, Fingerprint, Gauge, Globe2, Home as HomeIcon, LockKeyhole,
  Menu, Network, Radar, ScanFace, Search, Shield, ShieldCheck, Sparkles, UserRoundCheck, Volume2, VolumeX, X,
  Upload, Loader2, Image as ImageIcon,
} from "lucide-react";
import BorderScene from "@/components/three/BorderScene";
import GlobalGlobe from "@/components/three/GlobalGlobe";
import { checkpoints, type Checkpoint } from "@/data/checkpoints";
import { useScanSimulation, type ScanState } from "@/hooks/useScanSimulation";
import { toast } from "sonner";

type Mode = "HOME" | "DOCUMENT_SCAN" | "LIVE_VERIFICATION" | "GLOBAL_INTELLIGENCE" | "RISK_ANALYSIS" | "DATABASE" | "ANALYTICS" | "CASE_RECORDS";

type NavItem = { label: string; mode: Mode; icon: typeof HomeIcon };

const menuItems: NavItem[] = [
  { label: "Home", mode: "HOME", icon: HomeIcon },
  { label: "Document Scan", mode: "DOCUMENT_SCAN", icon: FileScan },
  { label: "Live Verification", mode: "LIVE_VERIFICATION", icon: ScanFace },
  { label: "Intelligence Map", mode: "GLOBAL_INTELLIGENCE", icon: Globe2 },
  { label: "Risk Analysis", mode: "RISK_ANALYSIS", icon: AlertTriangle },
  { label: "Database", mode: "DATABASE", icon: Database },
  { label: "Analytics", mode: "ANALYTICS", icon: BarChart3 },
  { label: "Case Records", mode: "CASE_RECORDS", icon: BookOpenCheck },
];

const processSteps = [
  { number: "01", title: "OCR extraction", detail: "Reading identity data", icon: FileScan, tone: "cyan" },
  { number: "02", title: "Document validation", detail: "Checking official standards", icon: ShieldCheck, tone: "green" },
  { number: "03", title: "Tampering detection", detail: "Inspecting security layers", icon: Search, tone: "amber" },
  { number: "04", title: "Face verification", detail: "Matching identity in real time", icon: ScanFace, tone: "cyan" },
];

const scanLabels: Record<ScanState, string> = {
  IDLE: "READY FOR DOCUMENT", DOCUMENT_DETECTED: "DOCUMENT DETECTED", OCR_EXTRACTION: "OCR EXTRACTION",
  DOCUMENT_VALIDATION: "DOCUMENT VALIDATION", TAMPERING_ANALYSIS: "TAMPERING ANALYSIS", FACE_VERIFICATION: "FACE VERIFICATION",
  RISK_ANALYSIS: "RISK ANALYSIS", VERIFIED: "IDENTITY VERIFIED",
};

const riskFactors = [
  ["Document integrity", "98%", "good"], ["Identity match", "98.7%", "good"], ["Expiry status", "VALID", "good"],
  ["Database status", "CLEAR", "good"], ["Travel history", "LOW SIGNAL", "neutral"], ["Anomaly score", "4%", "good"],
];

const cases = [
  { id: "BG-20481", reason: "Photo manipulation detected", risk: "87", status: "Review required", tone: "high", checkpoint: "T3 · New Delhi" },
  { id: "BG-20480", reason: "Identity mismatch", risk: "64", status: "Escalated", tone: "medium", checkpoint: "T2 · Mumbai" },
  { id: "BG-20479", reason: "Document expired", risk: "42", status: "Review required", tone: "medium", checkpoint: "T1 · Amritsar" },
  { id: "BG-20478", reason: "All checks passed", risk: "08", status: "Verified", tone: "low", checkpoint: "T3 · New Delhi" },
];

// Three-way outcome: an image may be a genuine document, a tampered one, or
// not an identity document at all. Only the first two are tampering-related.
function verdictText(r: ScanResult | null) {
  if (r && r.isDocument === false) {
    return {
      badge: "INVALID",
      title: "INVALID DOCUMENT",
      headline: "INVALID DOCUMENT",
      detail: "No identity document detected — cannot verify",
      tone: "#ff7068",
      warn: true,
    };
  }
  if (r && r.isTampered) {
    return {
      badge: "REVIEW",
      title: "TAMPERING DETECTED",
      headline: "REVIEW REQUIRED",
      detail: `Risk ${r.riskLevel} — officer review recommended`,
      tone: "#ff7068",
      warn: true,
    };
  }
  return {
    badge: "MATCH",
    title: "DOCUMENT AUTHENTIC",
    headline: "DOCUMENT AUTHENTIC",
    detail: "No significant tampering detected",
    tone: "#5cf298",
    warn: false,
  };
}

function modeTitle(mode: Mode) {
  return { HOME: "Command center", DOCUMENT_SCAN: "Document scan", LIVE_VERIFICATION: "Live verification", GLOBAL_INTELLIGENCE: "Global intelligence", RISK_ANALYSIS: "Risk analysis", DATABASE: "Database validation", ANALYTICS: "Analytics command", CASE_RECORDS: "Case records" }[mode];
}

interface ScanResult {
  authenticityScore: number;
  isTampered: boolean;
  isDocument?: boolean;
  ocrText: string;
  riskLevel: "Low" | "Medium" | "High";
  decision: "Verify" | "Review" | "Reject";
  status: "complete" | "error";
  source: string;
  // Detailed fields from new backend (optional)
  overall_score?: number;
  officer_decision?: string;
  extracted_fields?: any;
  mrz_validation?: boolean;
  tampering_detected?: boolean;
  heatmap_mean?: number;
  xai_breakdown?: any[];
  heuristic_features?: any;
  details?: any;
  results?: any;
}

interface UploadState {
  isScanning: boolean;
  scanResult: ScanResult | null;
  selectedFile: File | null;
  previewUrl: string | null;
  fileName: string | null;
  fileSize: number | null;
  fileKey: number; // force update
}

export default function Home() {
  const scan = useScanSimulation();
  const [activeMode, setActiveMode] = useState<Mode>(() => {
    const requestedMode = new URLSearchParams(window.location.search).get("mode");
    return requestedMode === "GLOBAL_INTELLIGENCE" ? "GLOBAL_INTELLIGENCE" : "HOME";
  });
  const [mobileNav, setMobileNav] = useState(false);
  const [introVisible, setIntroVisible] = useState(true);
  const [soundOn, setSoundOn] = useState(false);
  const [forensicMode, setForensicMode] = useState("ORIGINAL");
  const [selectedCase, setSelectedCase] = useState(cases[0]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<Checkpoint>(checkpoints[0]);
  const [showAlert, setShowAlert] = useState(true);
  const [testCounter, setTestCounter] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadState, setUploadState] = useState<UploadState>({
    isScanning: false,
    scanResult: null,
    selectedFile: null,
    previewUrl: null,
    fileName: null,
    fileSize: null,
    fileKey: 0,
  });

  // toast from sonner is used directly

  useEffect(() => {
    const timer = window.setTimeout(() => setIntroVisible(false), 1500);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    const handlePointerMove = (event: PointerEvent) => {
      document.documentElement.style.setProperty("--cursor-x", `${event.clientX}px`);
      document.documentElement.style.setProperty("--cursor-y", `${event.clientY}px`);
    };
    window.addEventListener("pointermove", handlePointerMove);
    return () => window.removeEventListener("pointermove", handlePointerMove);
  }, []);

  const fileRef = useRef<File | null>(null);
  const currentTime = useMemo(() => new Intl.DateTimeFormat("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false, timeZone: "Asia/Kolkata" }).format(new Date()), []);
  const currentLabel = scanLabels[scan.scanState];

  useEffect(() => {
    (window as any).debugUploadState = uploadState;
    (window as any).debugHasFile = !!uploadState.selectedFile || !!uploadState.fileName;
    (window as any).debugTestCounter = testCounter;
  }, [uploadState, testCounter]);

  const enterCommandCenter = () => { setIntroVisible(false); setActiveMode("HOME"); };
  const startDocumentScan = () => {
    setIntroVisible(false);
    setActiveMode("DOCUMENT_SCAN");
    // Reset only — NOT scan.startScan(). Starting the animation here left
    // scan.isRunning stuck true, which disabled the "Verify / Scan" button
    // forever (performScan starts the animation itself).
    scan.resetScan();
  };
  const handleNav = (item: NavItem) => { setActiveMode(item.mode); setMobileNav(false); setShowAlert(true); };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    console.log("[handleFileSelect] triggered", e.target.files?.[0]?.name, e.target.files?.[0]?.type);
    const file = e.target.files?.[0];
    if (file) {
      console.log("[handleFileSelect] file", file.name, file.type, file.size);
      // Validate type
      if (!["image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"].includes(file.type) && !file.name.match(/\.(png|jpg|jpeg|webp|bmp|tiff)$/i)) {
        console.log("[handleFileSelect] invalid type");
        toast("Invalid file type", { description: "Please upload PNG, JPG, or JPEG" });
        return;
      }
      if (file.size > 16 * 1024 * 1024) {
        toast("File too large", { description: "Max 16MB allowed" });
        return;
      }
      const preview = URL.createObjectURL(file);
      fileRef.current = file;
      // A previous scan leaves scan.isRunning true, which keeps the
      // "Verify / Scan" button stuck on "Scanning…". Picking a new file
      // starts over, so reset the animation (performScan starts it itself).
      scan.resetScan();
      flushSync(() => {
        setUploadState({ isScanning: false, scanResult: null, selectedFile: file, previewUrl: preview, fileName: file.name, fileSize: file.size, fileKey: Date.now() });
      });
      flushSync(() => {
        setTestCounter(c => c+1);
      });
      // Note: Do not auto-switch mode here to avoid batching conflict; user already in correct mode or will switch manually
      // setActiveMode("DOCUMENT_SCAN");
    } else {
      console.log("[handleFileSelect] no file");
    }
  };

  const clearFileInput = () => {
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const performScan = async () => {
    const file = fileRef.current || uploadState.selectedFile;
    console.log("[performScan] file", file?.name, "ref", fileRef.current?.name, "state", uploadState.selectedFile?.name, "fileName", uploadState.fileName);
    if (!file) {
      console.log("[performScan] no file");
      toast("No document selected", { description: "Please upload an image first" });
      return;
    }
    console.log("[performScan] startScan animation");
    scan.startScan();
    console.log("[performScan] set isScanning true");
    flushSync(() => {
      setUploadState(prev => ({ ...prev, isScanning: true, scanResult: null, fileName: prev.fileName, fileSize: prev.fileSize, fileKey: prev.fileKey }));
    });

    // Advance scan steps at realistic intervals while the API call runs
    const stepTimers: number[] = [];
    const stepDelays = [800, 1600, 2800, 4200, 5400]; // OCR -> Validation -> Tampering -> Face -> Risk
    stepDelays.forEach((delay) => {
      stepTimers.push(window.setTimeout(() => scan.advance(), delay));
    });
    
    try {
      console.log("[performScan] creating FormData", file.name, file.size);
      const formData = new FormData();
      formData.append("document", file);
      console.log("[performScan] FormData created");

      const endpoint = "/api/analyze-document";
      let response: Response;
      try {
        console.log("[performScan] fetching", endpoint);
        response = await fetch(endpoint, {
          method: "POST",
          body: formData,
        });
        console.log("[performScan] fetch response", response.status, response.headers.get("content-type"));
        const contentType = response.headers.get("content-type") || "";
        if (!response.ok && response.status === 404) throw new Error("proxy 404");
        if (!contentType.includes("json") && !response.ok) throw new Error("not json");
      } catch (proxyErr: any) {
        console.log("[performScan] proxyErr", proxyErr?.message, "fallback to Flask");
        response = await fetch("http://localhost:5000/api/analyze-document", {
          method: "POST",
          body: formData,
        });
        console.log("[performScan] fallback response", response.status);
      }

      if (!response.ok) {
        const errText = await response.text().catch(() => "");
        throw new Error(`Backend error: ${response.status} ${errText.slice(0,120)}`);
      }

      console.log("[performScan] awaiting json");
      const data = await response.json();
      console.log("[performScan] data received", JSON.stringify(data).slice(0,300));
      let result: ScanResult;
      if (data.results && typeof data.results.overall_score === "number") {
        result = {
          authenticityScore: data.authenticityScore ?? Math.round(data.results.overall_score),
          isTampered: data.isTampered ?? data.results.tampering_detected ?? false,
          isDocument: data.isDocument ?? data.results?.is_document ?? true,
          ocrText: data.ocrText ?? data.results.ocr_text ?? "",
          riskLevel: data.riskLevel ?? (data.results.risk_level === "GENUINE" ? "Low" : data.results.risk_level === "SUSPICIOUS" ? "Medium" : "High"),
          decision: data.decision ?? (data.results.risk_level === "GENUINE" ? "Verify" : "Review"),
          status: "complete",
          source: data.source ?? "sih26188-pipeline",
          details: data.details ?? data.results,
          results: data.results,
          overall_score: data.results.overall_score,
          xai_breakdown: data.results.xai_breakdown,
          heuristic_features: data.results.heuristic_features,
        };
      } else if (data.authenticityScore !== undefined) {
        result = data as ScanResult;
      } else if (data.success && data.results) {
        const r = data.results;
        result = {
          authenticityScore: r.overall_score ?? 50,
          isTampered: r.tampering_detected ?? false,
          isDocument: r.is_document ?? true,
          ocrText: r.ocr_text ?? Object.values(r.extracted_fields ?? {}).join(" ") ?? "",
          riskLevel: r.risk_level === "GENUINE" ? "Low" : r.risk_level === "SUSPICIOUS" ? "Medium" : "High",
          decision: r.risk_level === "GENUINE" ? "Verify" : "Review",
          status: "complete",
          source: "sih26188-pipeline",
          details: r,
        };
      } else {
        result = data as ScanResult;
      }

      console.log("[performScan] result prepared", result.authenticityScore, result.decision);
      // Complete the scan animation
      scan.complete();
      flushSync(() => {
        setUploadState(prev => {
          console.log("[performScan] setUploadState with result", result.authenticityScore);
          return { ...prev, isScanning: false, scanResult: result, fileName: prev.fileName, fileSize: prev.fileSize, fileKey: prev.fileKey };
        });
      });
      console.log("[performScan] setUploadState called");

      toast(`${result.decision} — Score: ${result.authenticityScore}/100`, {
        description: result.isTampered ? "Tampering indicators detected — review required" : result.ocrText ? `OCR: ${result.ocrText.substring(0, 60)}...` : "Document processed successfully",
      });

    } catch (error: any) {
      console.log("[performScan] catch error", error?.message);
      console.error("Scan failed:", error);
      scan.fail();
      toast("Scan failed", {
        // @ts-ignore sonner variant prop
        variant: "destructive",
        description: error.message || "Unable to connect to backend. Ensure Flask is running on port 5000 (python app.py)",
      });
      setUploadState(prev => ({ ...prev, isScanning: false, fileName: prev.fileName, fileSize: prev.fileSize, fileKey: prev.fileKey }));
    } finally {
      // Clean up step timers
      stepTimers.forEach((t) => window.clearTimeout(t));
    }
  };

  const resetUpload = () => {
    setUploadState(prev => {
      if (prev.previewUrl) URL.revokeObjectURL(prev.previewUrl);
      fileRef.current = null;
      return {
        isScanning: false,
        scanResult: null,
        selectedFile: null,
        previewUrl: null,
        fileName: null,
        fileSize: null,
        fileKey: 0,
      };
    });
    scan.resetScan();
    clearFileInput();
  };

  const triggerFilePicker = () => {
    fileInputRef.current?.click();
  };

  const v = verdictText(uploadState.scanResult);

  return (
    <main className={`borderguard-shell ${introVisible ? "intro-active" : ""}`}>
      <div className="hud-cursor" aria-hidden="true" />
      <div className="ambient-grid" aria-hidden="true" />
        {introVisible && <section className="intro-screen"><div className="intro-grid" /><div className="intro-point" /><div className="intro-content"><div className="intro-mark"><Shield size={31} /><span /></div><div className="intro-wordmark">BORDER<span>GUARD</span> AI</div><div className="intro-rule" /><p>AI-POWERED IDENTITY & DOCUMENT INTELLIGENCE</p><div className="intro-actions"><button className="hud-button hud-button-primary" onClick={startDocumentScan}><ScanFace size={15} /> Start document scan</button><button className="hud-button" onClick={enterCommandCenter}><Radar size={15} /> Enter command center</button></div><span className="intro-demo">SIH 2026 — AI Document Authenticity System</span></div></section>}

      <header className="topbar">
        <button className="mobile-menu-button" aria-label="Open navigation" onClick={() => setMobileNav((value) => !value)}>{mobileNav ? <X size={18} /> : <Menu size={18} />}</button>
        <div className="brand-lockup"><div className="brand-mark"><Shield size={24} strokeWidth={1.5} /><span className="brand-node" /></div><div><div className="wordmark">BORDER<span>GUARD</span> AI</div><div className="brand-tagline">Secure borders · safer tomorrow</div></div></div>
        <div className="topbar-motto"><span>People move the world</span><strong>We keep it safe</strong></div>
        <div className="ministry-block"><div className="ministry-emblem"><ShieldCheck size={18} /></div><div className="ministry-copy"><strong>Ministry of Home Affairs</strong><span>Document Intelligence System</span></div><div className="time-block"><strong>{currentTime}</strong><span>{new Date().toLocaleDateString('en-IN', { weekday: 'short', day: '2-digit', month: 'short', year: 'numeric' })}</span></div></div>
      </header>

      <aside className={`sidebar ${mobileNav ? "sidebar-open" : ""}`}>
        <div className="sidebar-heading"><span className="live-dot" />CONTROL DESK</div>
        <nav aria-label="Primary navigation">{menuItems.map((item) => { const Icon = item.icon; return <button key={item.mode} className={`nav-item ${activeMode === item.mode ? "nav-item-active" : ""}`} onClick={() => handleNav(item)}><Icon size={17} strokeWidth={1.6} /><span>{item.label}</span>{activeMode === item.mode && <ChevronRight className="nav-chevron" size={14} />}</button>; })}</nav>
        <div className="sidebar-footer"><div className="shield-orbit"><Globe2 size={37} strokeWidth={1.2} /><span /></div><p>Protect people.<br />Preserve possibilities.</p><div className="system-status"><CircleDot size={12} /> All systems operational</div><div className="demo-badge"><LockKeyhole size={12} /> SIH 2026</div></div>
      </aside>

      <section className="workspace">
        <div className="workspace-intro"><div><div className="eyebrow"><span className="eyebrow-line" />{activeMode === "HOME" ? "ACTIVE INSPECTION / 07" : `COMMAND MODE / ${activeMode}`}</div><h1>{modeTitle(activeMode)} <em>{activeMode === "HOME" ? "intelligence" : ""}</em></h1></div><div className="workspace-actions"><button className={`sound-toggle ${soundOn ? "sound-on" : ""}`} aria-label="Toggle sound" onClick={() => setSoundOn((value) => !value)}>{soundOn ? <Volume2 size={16} /> : <VolumeX size={16} />}<span>{soundOn ? "SOUND ON" : "SOUND OFF"}</span></button><button className="icon-button" aria-label="Search cases"><Search size={17} /></button><button className="icon-button notification-button" aria-label="Notifications"><Bell size={17} /><span /></button><button className="operator-avatar" aria-label="Operator profile">AS</button></div></div>
        {activeMode === "HOME" || activeMode === "DOCUMENT_SCAN" || activeMode === "LIVE_VERIFICATION"
          ? <DashboardMode scan={scan} currentLabel={currentLabel} activeMode={activeMode} startDocumentScan={startDocumentScan} uploadState={uploadState} onFileSelect={handleFileSelect} onScan={performScan} onResetUpload={resetUpload} onPickFile={triggerFilePicker} fileInputRef={fileInputRef} />
          : activeMode === "GLOBAL_INTELLIGENCE" ? <GlobeMode selectedCheckpoint={selectedCheckpoint} onSelectCheckpoint={setSelectedCheckpoint} /> : <OperationalMode mode={activeMode} forensicMode={forensicMode} setForensicMode={setForensicMode} selectedCase={selectedCase} setSelectedCase={setSelectedCase} scanResult={uploadState.scanResult} />}
      </section>

      <aside className="evidence-column">
        <div className="evidence-heading"><div><span className="eyebrow-line" />IDENTITY EVIDENCE</div><span className="case-id">DOCUMENT ANALYSIS</span></div>
        {/* Dynamic evidence card — shows real backend result when available, otherwise demo */}
        {uploadState.scanResult ? (
          <section className="identity-card panel-surface">
            <div className="identity-card-top">
              <span className="verified-label" style={{ color: v.tone }}>
                <span style={{ background: v.tone }} />{v.badge}
              </span>
              <Fingerprint size={18} />
            </div>
            <div className="identity-portrait-row">
              <div className="portrait-frame">
                {uploadState.previewUrl ? (
                  <img src={uploadState.previewUrl} alt="Uploaded document" style={{ width: "100%", height: "100%", objectFit: "cover", position: "absolute", inset: 0 }} />
                ) : (
                  <div className="portrait-placeholder"><UserRoundCheck size={42} strokeWidth={1} /></div>
                )}
                <span className="portrait-scanline" />
              </div>
              <div className="face-match">
                <div className="face-wire"><ScanFace size={53} strokeWidth={1} /></div>
                <strong style={{ color: v.tone }}>{uploadState.scanResult.authenticityScore}%</strong><span>AUTH SCORE</span>
              </div>
            </div>
            <div className="identity-name"><span>BACKEND RESULT · {uploadState.scanResult.source}</span><strong style={{ color: v.tone }}>{v.headline}</strong></div>
            <div className="identity-grid">
              <div><span>Risk level</span><strong style={{ color: uploadState.scanResult.riskLevel === "High" ? "#ff7068" : uploadState.scanResult.riskLevel === "Medium" ? "#f1bb66" : "#5cf298" }}>{uploadState.scanResult.riskLevel.toUpperCase()}</strong></div>
              <div><span>Decision</span><strong>{uploadState.scanResult.decision.toUpperCase()}</strong></div>
              <div><span>OCR text</span><strong style={{ fontSize: "8px", lineHeight: "1.4" }}>{uploadState.scanResult.ocrText ? uploadState.scanResult.ocrText.slice(0, 60) + (uploadState.scanResult.ocrText.length > 60 ? "…" : "") : "No text"}</strong></div>
              <div><span>Source</span><strong>SIH26188</strong></div>
            </div>
            <div className={`authenticity-box ${v.warn ? "authenticity-warn" : "authenticity-success"}`}>
              <div className="authenticity-icon">{v.warn ? <AlertTriangle size={20} /> : <Check size={20} />}</div>
              <div>
                <strong>{v.title}</strong>
                <span>{v.detail}</span>
              </div>
            </div>
            <div className="risk-block">
              <div className="risk-label"><span>RISK SCORE</span><strong style={{ color: uploadState.scanResult.riskLevel === "High" ? "#ff7068" : uploadState.scanResult.riskLevel === "Medium" ? "#f1bb66" : "#5cf298" }}>{100 - uploadState.scanResult.authenticityScore} <small>/ 100</small></strong></div>
              <div className="risk-meter"><span style={{ width: `${100 - uploadState.scanResult.authenticityScore}%`, background: uploadState.scanResult.riskLevel === "High" ? "#ff7068" : uploadState.scanResult.riskLevel === "Medium" ? "#f1bb66" : "#5cf298" }} /></div>
              <div className="risk-caption"><span>Low risk</span><span>High risk</span></div>
            </div>
            {uploadState.scanResult.details?.xai_breakdown && (
              <div style={{ marginTop: 12, borderTop: "1px solid rgba(151,204,227,.13)", paddingTop: 12 }}>
                <div style={{ font: "8px IBM Plex Mono, monospace", color: "#6e909d", letterSpacing: ".12em", marginBottom: 8 }}>EXPLAINABILITY (XAI)</div>
                {uploadState.scanResult.details.xai_breakdown.slice(0, 3).map((item: any, i: number) => (
                  <div key={i} style={{ display: "flex", justifyContent: "space-between", font: "9px IBM Plex Mono, monospace", color: "#b6cad5", marginBottom: 4 }}>
                    <span>{item.feature}</span><span style={{ color: item.contribution > 50 ? "#5cf298" : "#f1bb66" }}>{item.contribution}%</span>
                  </div>
                ))}
              </div>
            )}
          </section>
        ) : (
          <section className="identity-card panel-surface">
            <div className="identity-card-top"><span className="verified-label" style={{ color: "#6e909d" }}><span style={{ background: "#6e909d" }} />PENDING</span><Fingerprint size={18} /></div>
            <div className="identity-portrait-row"><div className="portrait-frame"><div className="portrait-placeholder"><UserRoundCheck size={42} strokeWidth={1} /></div><span className="portrait-scanline" /></div>
              <div className="face-match"><div className="face-wire"><ScanFace size={53} strokeWidth={1} /></div><strong style={{ color: "#6e909d" }}>—</strong><span>AUTH SCORE</span></div></div>
            <div className="identity-name"><span>NO DOCUMENT SCANNED</span><strong>Upload to begin analysis</strong></div>
            <div className="identity-grid"><div><span>Risk level</span><strong>—</strong></div><div><span>Decision</span><strong>—</strong></div><div><span>OCR text</span><strong>—</strong></div><div><span>Source</span><strong>SIH26188</strong></div></div>
            <div className="authenticity-box authenticity-success"><div className="authenticity-icon"><Check size={20} /></div><div><strong>READY TO VERIFY</strong><span>Upload a passport or ID image to start</span></div></div>
            <div className="risk-block"><div className="risk-label"><span>RISK SCORE</span><strong>— <small>/ 100</small></strong></div><div className="risk-meter"><span style={{ width: "0%" }} /></div><div className="risk-caption"><span>Low risk</span><span>High risk</span></div></div>
          </section>
        )}

        <button className={`verify-button ${uploadState.scanResult ? "is-verified" : scan.isVerified ? "is-verified" : ""}`} onClick={uploadState.scanResult || scan.isVerified ? resetUpload : startDocumentScan}>
          {uploadState.scanResult ? <><Check size={17} /> New verification</> : scan.isVerified ? <><Check size={17} /> Verification complete</> : <><Sparkles size={17} /> Start verification</>}
        </button>

        {showAlert && <div className="toast-message" role="status"><div><Sparkles size={15} /><span>{activeMode === "HOME" ? "Pipeline is up to date — ready for uploads." : `${modeTitle(activeMode)} loaded.`}</span></div><button aria-label="Dismiss" onClick={() => setShowAlert(false)}><X size={14} /></button></div>}

        <div className="evidence-meta" style={{ marginTop: 14, padding: "12px", border: "1px solid rgba(151,204,227,.13)", borderRadius: 8, background: "rgba(7,31,43,.64)" }}>
          <div style={{ font: "8px IBM Plex Mono, monospace", color: "#6e909d", letterSpacing: ".12em", marginBottom: 8 }}>UPLOAD FLOW</div>
          <div style={{ font: "9px IBM Plex Mono, monospace", color: "#b6cad5", lineHeight: "1.6" }}>
            Frontend → Upload → Verify/Scan → <strong style={{ color: "#42d9ff" }}>Flask /api/analyze-document</strong> → Real result → UI
            <br /><span style={{ color: "#5cf298" }}>No terminal input required</span>
          </div>
        </div>

        <footer className="shell-footer"><span>SECURE PROCESSING</span><i /><span>AUDIT LOG ENABLED</span><i /><span>SIH 2026</span><span className="footer-mark">For a safer world <b /></span></footer>
      </aside>
    </main>
  );
}

// ── DashboardMode: retains Three.js scene and scan animation, adds real upload wiring ──
function DashboardMode({ scan, currentLabel, activeMode, startDocumentScan, uploadState, onFileSelect, onScan, onResetUpload, onPickFile, fileInputRef }: any) {
  const isScanning = uploadState.isScanning || scan.isRunning;
  const hasFile = !!uploadState.selectedFile || !!uploadState.fileName;
  const hasResult = !!uploadState.scanResult;
  const notADocument = uploadState.scanResult?.isDocument === false;

  return (
    <div className="workspace-grid" style={{ display: "grid" }}>
      {/* Hidden file input — reused across renders (offscreen, not display:none for fileChooser compatibility) */}
      <input ref={fileInputRef} type="file" accept=".png,.jpg,.jpeg,.webp,.bmp,.tiff" onChange={onFileSelect} style={{ position: "absolute", width: 1, height: 1, opacity: 0, overflow: "hidden", clip: "rect(0,0,0,0)" }} />

      <div className="scan-stage panel-surface" style={{ gridColumn: "1" }}>
        <div className="stage-image" />
        <div className="stage-overlay" />
        <div className="stage-topline"><span className="stage-location"><Globe2 size={14} /> T3 · New Delhi · Live checkpoint</span><span className="secure-chip"><ShieldCheck size={14} /> SECURE PROCESSING</span></div>
        <div className="scene-shell" style={{ opacity: 0.96 }}>
          <BorderScene scanState={scan.scanState} />
          {/* Fallback scanner visible while Three.js loads or as overlay */}
          <div className="scene-fallback" style={{ display: "none" }} aria-hidden="true">
            <div className="fallback-scanner"><i /></div>
            <div className="fallback-passport">DOCUMENT<span>AWAITING SCAN</span></div>
            <div className="fallback-hologram"><ScanFace size={28} /> HOLOGRAM</div>
          </div>
        </div>
        <div className="scan-beam" style={{ opacity: isScanning ? 1 : 0.45 }} />
        <span className="scan-corner scan-corner-tl" />
        <span className="scan-corner scan-corner-tr" />
        <span className="scan-corner scan-corner-bl" />
        <span className="scan-corner scan-corner-br" />
        <div className="scan-panel-content">
          <div className="scan-status-row"><span className="live-dot" style={{ background: isScanning ? "#42d9ff" : "#5cf298" }} />{currentLabel}<strong>{scan.progress}%</strong></div>
          <div className="progress-track"><span style={{ width: `${scan.progress}%` }} /></div>
          <div className="hologram-wrap">
            {uploadState.previewUrl ? (
              <div style={{ position: "relative", zIndex: 2, width: "min(82%, 360px)", border: "1px solid rgba(66,217,255,.6)", borderRadius: 6, overflow: "hidden", background: "#06111a", boxShadow: "0 20px 22px rgba(0,0,0,.65)" }}>
                <img src={uploadState.previewUrl} alt="Preview" style={{ width: "100%", display: "block", maxHeight: 220, objectFit: "contain" }} />
                <div style={{ padding: "8px 10px", display: "flex", justifyContent: "space-between", alignItems: "center", background: "rgba(7,31,43,.9)", borderTop: "1px solid rgba(66,217,255,.2)" }}>
                  <span style={{ font: "8px IBM Plex Mono, monospace", color: "#b6cad5", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 160 }}>{uploadState.fileName || uploadState.selectedFile?.name}</span>
                  <span style={{ font: "7px IBM Plex Mono, monospace", color: "#5cf298" }}>{uploadState.fileSize ? (uploadState.fileSize / 1024).toFixed(0) : (uploadState.selectedFile ? (uploadState.selectedFile.size / 1024).toFixed(0) : "0")} KB</span>
                </div>
              </div>
            ) : (
              <div className="passport-visual" style={{ border: "1px dashed rgba(66,217,255,.35)", borderRadius: 8, minHeight: 140, display: "grid", placeItems: "center", background: "rgba(7,31,43,.55)", padding: 16, textAlign: "center" }}>
                <div style={{ color: "#b6cad5" }}>
                  <ImageIcon size={28} style={{ margin: "0 auto 8px", color: "#42d9ff" }} />
                  <div style={{ font: "10px IBM Plex Mono, monospace", letterSpacing: ".12em", color: "#eaf9fd" }}>NO DOCUMENT SELECTED</div>
                  <div style={{ font: "8px IBM Plex Mono, monospace", color: "#6e909d", marginTop: 6 }}>Upload a passport / ID image to start</div>
                </div>
              </div>
            )}
          </div>
          <div className="document-caption"><strong>CASE 7F-2048</strong><span>T3 · NEW DELHI</span></div>
        </div>
        <div className="stage-bottomline"><span><Clock3 size={13} /> T3 · 14:32 IST · Queue 07</span><span><Network size={13} /> PIPELINE: <strong className="mono" style={{ color: "#42d9ff" }}>{scan.label}</strong></span></div>
      </div>

      <div className="checklist-panel panel-surface" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <div className="panel-kicker"><span>VERIFICATION PIPELINE</span><span>{isScanning ? "SCANNING…" : hasResult ? "COMPLETE" : "IDLE"}</span></div>
        <div className="checklist-line" />
        <div className="checklist-list">
          {processSteps.map((step) => {
            const Icon = step.icon;
            const toneClass = step.tone === "green" ? "process-green" : step.tone === "amber" ? "process-amber" : "";
            // Determine active state based on scan
            const isActive = (step.number === "01" && scan.scanState === "OCR_EXTRACTION") || (step.number === "02" && scan.scanState === "DOCUMENT_VALIDATION") || (step.number === "03" && scan.scanState === "TAMPERING_ANALYSIS") || (step.number === "04" && scan.scanState === "FACE_VERIFICATION");
            return (
              <div key={step.number} className="check-row">
                <span className={`check-status ${isActive ? "check-status-active" : ""}`}>{isActive ? <Loader2 size={10} className="animate-spin" /> : <Check size={10} />}</span>
                <div>
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}><span style={{ font: "9px IBM Plex Mono, monospace", color: "#6e909d" }}>{step.number}</span><strong style={{ fontSize: 10, color: isActive ? "#42d9ff" : "#dbeef2" }}>{step.title}</strong></div>
                  <div style={{ fontSize: 9, color: "#72919c" }}>{step.detail}</div>
                </div>
                <div className="check-state">{isActive ? "ANALYZING…" : hasResult ? "COMPLETE" : "READY"}</div>
              </div>
            );
          })}
        </div>

        {/* ——— Real upload controls ——— */}
        <div style={{ marginTop: "auto", display: "grid", gap: 8, paddingTop: 14, borderTop: "1px solid rgba(151,204,227,.13)" }}>
          <button
            onClick={onPickFile}
            disabled={isScanning}
            className="hud-button"
            style={{ width: "100%", justifyContent: "center", opacity: isScanning ? 0.6 : 1 }}
          >
            <Upload size={14} /> {hasFile ? "Change document" : "Upload document"}
          </button>

          {hasFile && !hasResult && (
            <button
              onClick={onScan}
              disabled={isScanning}
              className="hud-button hud-button-primary"
              style={{ width: "100%", justifyContent: "center", opacity: isScanning ? 0.8 : 1 }}
            >
              {isScanning ? <><Loader2 size={14} className="animate-spin" /> Scanning…</> : <><Sparkles size={14} /> Verify / Scan</>}
            </button>
          )}

          {hasResult && (
            <button onClick={onResetUpload} className="hud-button" style={{ width: "100%", justifyContent: "center" }}>
              <X size={14} /> Clear & new scan
            </button>
          )}

          {hasFile && (
            <div style={{ font: "7px IBM Plex Mono, monospace", color: "#6e909d", textAlign: "center", lineHeight: "1.5" }}>
              Selected: {uploadState.fileName || uploadState.selectedFile?.name} · {uploadState.fileSize ? (uploadState.fileSize/1024).toFixed(0) : (uploadState.selectedFile ? (uploadState.selectedFile.size/1024).toFixed(0) : "0")} KB
              <br />Will POST to <span style={{ color: "#42d9ff" }}>/api/analyze-document</span> (Flask) {uploadState.fileKey ? `· key ${uploadState.fileKey}` : ""}
            </div>
          )}

          {/* Real backend result — replaces fake demo numbers */}
          {hasResult && (
            <div style={{ marginTop: 4, padding: 10, borderRadius: 8, background: notADocument || uploadState.scanResult.isTampered ? "rgba(255,112,104,.08)" : "rgba(92,242,152,.08)", border: `1px solid ${notADocument || uploadState.scanResult.isTampered ? "rgba(255,112,104,.3)" : "rgba(92,242,152,.3)"}` }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ font: "8px IBM Plex Mono, monospace", letterSpacing: ".12em", color: notADocument || uploadState.scanResult.isTampered ? "#ff7068" : "#5cf298" }}>{notADocument ? "INVALID DOCUMENT" : uploadState.scanResult.isTampered ? "REVIEW REQUIRED" : "VERIFIED"}</span>
                <strong style={{ font: "14px IBM Plex Mono, monospace", color: "#eaf9fd" }}>{uploadState.scanResult.authenticityScore}/100</strong>
              </div>
              <div style={{ font: "8px IBM Plex Mono, monospace", color: "#6e909d", marginTop: 6, lineHeight: "1.5", wordBreak: "break-word" }}>
                {uploadState.scanResult.ocrText ? `OCR: ${uploadState.scanResult.ocrText.slice(0, 80)}…` : "Processed successfully"}
              </div>
              <div style={{ font: "7px IBM Plex Mono, monospace", color: "#557986", marginTop: 6 }}>
                Risk: {uploadState.scanResult.riskLevel} · {uploadState.scanResult.decision} · {uploadState.scanResult.source}
              </div>
            </div>
          )}
        </div>

        <div className="pipeline-footer"><span>SECURE PROCESSING</span><strong>AUDIT LOG ENABLED</strong></div>
      </div>

      <div className="process-strip" style={{ gridColumn: "1 / -1" }}>
        {processSteps.map((step) => {
          const Icon = step.icon;
          return (
            <div key={step.number} className={`process-card ${step.tone === "green" ? "process-green" : step.tone === "amber" ? "process-amber" : ""}`}>
              <span className="process-number">{step.number}</span>
              <span className="process-icon"><Icon size={16} /></span>
              <span className="process-copy"><strong>{step.title}</strong><small>{step.detail}</small></span>
              <ChevronRight size={14} className="process-arrow" />
            </div>
          );
        })}
      </div>
    </div>
  );
}

function GlobeMode({ selectedCheckpoint, onSelectCheckpoint }: { selectedCheckpoint: Checkpoint; onSelectCheckpoint: (c: Checkpoint) => void }) {
  return (
    <div className="globe-mode">
      <div className="globe-panel panel-surface">
        <div className="panel-kicker"><span>GLOBAL INTELLIGENCE</span><span>SIH 2026</span></div>
        <div className="globe-instructions"><span><Radar size={12} /> Rotatable globe — drag to inspect</span><span className="globe-legend"><span className="live-dot" /> Live checkpoint</span></div>
        <GlobalGlobe selectedId={selectedCheckpoint.id} onSelect={onSelectCheckpoint} />
        <div className="globe-footer"><span><Network size={12} /> Routes: <strong>{checkpoints.length} checkpoints</strong></span><span><Gauge size={12} /> Intelligence feed</span></div>
      </div>
      <div className="checkpoint-panel panel-surface">
        <div className="panel-kicker"><span>CHECKPOINT</span><span>{selectedCheckpoint.id}</span></div>
        <h3 style={{ margin: "14px 0 6px", fontSize: 16, color: "#eaf9fd" }}>{selectedCheckpoint.name}</h3>
        <div style={{ font: "8px IBM Plex Mono, monospace", color: "#6e909d", marginBottom: 12 }}>{selectedCheckpoint.country} · {selectedCheckpoint.region}</div>
        <div className={`checkpoint-status checkpoint-status-${selectedCheckpoint.status.toLowerCase()}`}><span style={{ background: selectedCheckpoint.status === "CLEAR" ? "#5cf298" : selectedCheckpoint.status === "REVIEW" ? "#f1bb66" : "#ff7068" }} />{selectedCheckpoint.status}</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 16 }}>
          <div style={{ padding: 10, border: "1px solid rgba(151,204,227,.14)", borderRadius: 8, background: "rgba(7,31,43,.5)" }}><div style={{ font: "7px IBM Plex Mono, monospace", color: "#6e909d" }}>THROUGHPUT</div><strong style={{ font: "16px IBM Plex Mono, monospace", color: "#eaf9fd" }}>{selectedCheckpoint.throughput}</strong></div>
          <div style={{ padding: 10, border: "1px solid rgba(151,204,227,.14)", borderRadius: 8, background: "rgba(7,31,43,.5)" }}><div style={{ font: "7px IBM Plex Mono, monospace", color: "#6e909d" }}>RISK</div><strong style={{ font: "16px IBM Plex Mono, monospace", color: selectedCheckpoint.status === "ELEVATED" ? "#ff7068" : selectedCheckpoint.status === "REVIEW" ? "#f1bb66" : "#5cf298" }}>{selectedCheckpoint.status === "ELEVATED" ? "64" : selectedCheckpoint.status === "REVIEW" ? "42" : "08"}/100</strong></div>
        </div>
        <div className="checkpoint-note" style={{ marginTop: 12, padding: 10, background: "rgba(7,31,43,.4)", border: "1px solid rgba(151,204,227,.12)", borderRadius: 8, font: "9px IBM Plex Mono, monospace", color: "#8eb5c0", lineHeight: "1.6" }}>
          Coordinates: {selectedCheckpoint.lat.toFixed(2)}, {selectedCheckpoint.lon.toFixed(2)}<br />Intelligence data — border checkpoint monitoring.
        </div>
        <div style={{ marginTop: 14, font: "8px IBM Plex Mono, monospace", color: "#557986" }}>SELECTED CHECKPOINT</div>
        <div style={{ marginTop: 8, display: "grid", gap: 6 }}>
          {checkpoints.slice(0, 5).map((cp) => (
            <button key={cp.id} onClick={() => onSelectCheckpoint(cp)} className="checkpoint-list-row" style={{ textAlign: "left", padding: "8px 10px", border: `1px solid ${cp.id === selectedCheckpoint.id ? "rgba(66,217,255,.5)" : "rgba(151,204,227,.12)"}`, background: cp.id === selectedCheckpoint.id ? "rgba(66,217,255,.08)" : "rgba(7,31,43,.3)", borderRadius: 6, color: "#b6cad5", font: "9px IBM Plex Mono, monospace" }}>
              {cp.name} · {cp.status}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function OperationalMode({ mode, forensicMode, setForensicMode, selectedCase, setSelectedCase, scanResult }: any) {
  const hasResult = scanResult && scanResult.authenticityScore !== undefined;
  const score = hasResult ? scanResult.authenticityScore : 0;
  const risk = hasResult ? scanResult.riskLevel : "—";
  const riskScore = hasResult ? 100 - score : 0;
  const isTampered = hasResult ? scanResult.isTampered : false;
  const notDoc = hasResult ? scanResult.isDocument === false : false;
  const ocrText = hasResult ? (scanResult.ocrText || "") : "";
  const decision = hasResult ? scanResult.decision : "—";
  const fields = hasResult ? (scanResult.details?.extracted_fields || scanResult.extracted_fields || {}) : {};
  const riskColor = risk === "High" ? "#ff7068" : risk === "Medium" ? "#f1bb66" : "#5cf298";
  const factors = hasResult ? [
    ["Authenticity score", `${score}%`, score >= 65 ? "good" : "warn"],
    ["Risk level", risk, risk === "Low" ? "good" : "warn"],
    ["OCR text", ocrText ? `${ocrText.length} chars` : "None", ocrText ? "good" : "neutral"],
    ["Fields detected", `${Object.values(fields).filter((v: any) => v && String(v).trim()).length}`, "neutral"],
    ["Decision", decision, decision === "Verify" ? "good" : "warn"],
    ["Document status", notDoc ? "Not a document" : "Identity document", notDoc ? "warn" : "good"],
    ["Classifier confidence", hasResult ? `${Math.round((scanResult.details?.classifier_output?.confidence || 0) * 100)}%` : "—", "neutral"],
  ] : [
    ["Document integrity", "—", "neutral"], ["Identity match", "—", "neutral"], ["Expiry status", "—", "neutral"],
    ["Database status", "—", "neutral"], ["OCR extraction", "—", "neutral"], ["Anomaly score", "—", "neutral"],
  ];
  return (
    <div className="mode-grid">
      <div className="mode-panel panel-surface">
        <div className="panel-kicker"><span>{modeTitle(mode).toUpperCase()}</span><span>SYSTEM DATA</span></div>
        {mode === "RISK_ANALYSIS" && (
          <div className="gauge">
            <div className="gauge-ring" style={{ borderColor: riskColor, boxShadow: `0 0 20px ${riskColor}33` }}><strong style={{ color: riskColor }}>{hasResult ? riskScore : "—"}</strong><span>/100</span></div>
            <div className="gauge-label" style={{ color: riskColor }}>{hasResult ? `${risk.toUpperCase()} RISK` : "NO DATA"}</div>
            <div className="factor-list" style={{ width: "100%" }}>
              {factors.map(([label, value, tone]: any) => (
                <div key={label} className="factor-row"><span>{label}</span><strong className={tone === "good" ? "factor-good" : tone === "warn" ? "factor-warn" : "factor-neutral"}>{value}</strong><div className="factor-track"><i style={{ width: value.includes("%") ? value : "60%", background: tone === "good" ? "#5cf298" : tone === "warn" ? "#ff7068" : "#f1bb66" }} /></div></div>
              ))}
            </div>
          </div>
        )}
        {mode === "DATABASE" && (
          <div className="database-flow">
            <div className="core-node"><Boxes size={14} /><div><strong>Identity Core</strong><span>Unified registry</span></div></div>
            {["Passport Registry", "Visa Registry", "Watchlist", "Immigration Records"].map((name) => (
              <div key={name} className="database-node"><Database size={14} /><span>{name}</span><strong>ONLINE</strong><small>LIVE</small></div>
            ))}
            <div className="database-footer"><Check size={14} /> All connections active<span>Last sync: {new Date().toLocaleDateString()}</span></div>
          </div>
        )}
        {mode === "ANALYTICS" && (
          <div className="analytics-grid" style={{ gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            {[
              ["Throughput", "1,284", "+4.2%"],
              ["Avg. processing", "38s", "-1.1s"],
              ["Risk escalations", "07", "Stable"],
              ["System uptime", "99.8%", "Operational"],
            ].map(([label, value, delta]: any) => (
              <div key={label} className="metric-card panel-surface"><span>{label}</span><strong>{value}</strong><small>{delta}</small></div>
            ))}
          </div>
        )}
        {mode === "CASE_RECORDS" && (
          <div className="cases-panel">
            {hasResult && (
              <button onClick={() => setSelectedCase({ id: "SCAN-001", reason: notDoc ? "Not an identity document" : isTampered ? "Tampering detected" : "Document verified", risk: String(score), status: decision, tone: notDoc || isTampered ? "high" : "low", checkpoint: "Current Scan" })} style={{ width: "100%", textAlign: "left", padding: 12, marginBottom: 8, border: `1px solid ${riskColor}55`, background: `${riskColor}11`, borderRadius: 8 }}>
                <div style={{ display: "flex", justifyContent: "space-between", font: "9px IBM Plex Mono, monospace" }}><span style={{ color: "#6e909d" }}>LATEST SCAN</span><span style={{ color: riskColor }}>{score}/100</span></div>
                <div style={{ fontSize: 11, color: "#dbeef2", marginTop: 6 }}>{notDoc ? "Invalid document" : isTampered ? "Tampering detected" : "Document authentic"} — {decision}</div>
                <div style={{ font: "8px IBM Plex Mono, monospace", color: "#6e909d", marginTop: 4 }}>Risk: {risk} · {ocrText ? ocrText.slice(0, 40) + "…" : "No OCR"}</div>
              </button>
            )}
            {cases.map((c) => (
              <button key={c.id} onClick={() => setSelectedCase(c)} style={{ width: "100%", textAlign: "left", padding: 12, marginBottom: 8, border: `1px solid ${selectedCase.id === c.id ? "rgba(66,217,255,.4)" : "rgba(151,204,227,.12)"}`, background: selectedCase.id === c.id ? "rgba(66,217,255,.06)" : "rgba(7,31,43,.3)", borderRadius: 8 }}>
                <div style={{ display: "flex", justifyContent: "space-between", font: "9px IBM Plex Mono, monospace" }}><span style={{ color: "#6e909d" }}>{c.id}</span><span style={{ color: c.tone === "high" ? "#ff7068" : c.tone === "medium" ? "#f1bb66" : "#5cf298" }}>{c.risk}/100</span></div>
                <div style={{ fontSize: 11, color: "#dbeef2", marginTop: 6 }}>{c.reason}</div>
                <div style={{ font: "8px IBM Plex Mono, monospace", color: "#6e909d", marginTop: 4 }}>{c.checkpoint} · {c.status}</div>
              </button>
            ))}
          </div>
        )}
        {(mode !== "RISK_ANALYSIS" && mode !== "DATABASE" && mode !== "ANALYTICS" && mode !== "CASE_RECORDS") && (
          <div style={{ padding: "40px 20px", textAlign: "center", color: "#6e909d", font: "10px IBM Plex Mono, monospace" }}>
            <FileCheck2 size={28} style={{ margin: "0 auto 12px", color: "#42d9ff" }} />
            {modeTitle(mode)} — module ready.<br />Switch to Document Scan to test the upload flow.
          </div>
        )}
      </div>
      <div className="mode-panel panel-surface">
        <div className="panel-kicker"><span>CASE DETAIL</span><span>{selectedCase.id}</span></div>
        <h3 style={{ margin: "14px 0 6px", color: "#eaf9fd" }}>{selectedCase.reason}</h3>
        <div style={{ font: "8px IBM Plex Mono, monospace", color: "#6e909d" }}>{selectedCase.checkpoint} · {selectedCase.status}</div>
        <div style={{ display: "grid", gap: 12, marginTop: 16 }}>
          <div style={{ padding: 12, border: "1px solid rgba(151,204,227,.14)", borderRadius: 8, background: "rgba(7,31,43,.4)" }}>
            <div style={{ font: "8px IBM Plex Mono, monospace", color: "#6e909d", marginBottom: 6 }}>RISK SCORE</div>
            <strong style={{ font: "22px IBM Plex Mono, monospace", color: selectedCase.tone === "high" ? "#ff7068" : "#f1bb66" }}>{selectedCase.risk}/100</strong>
          </div>
          <div style={{ padding: 12, border: "1px solid rgba(151,204,227,.14)", borderRadius: 8, background: "rgba(7,31,43,.4)" }}>
            <div style={{ font: "8px IBM Plex Mono, monospace", color: "#6e909d", marginBottom: 6 }}>FORENSIC VIEW</div>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {["ORIGINAL", "EDGE MAP", "HEATMAP", "NOISE ANALYSIS"].map((m) => (
                <button key={m} onClick={() => setForensicMode(m)} className="hud-button" style={{ minHeight: 28, padding: "0 10px", fontSize: 7, background: forensicMode === m ? "rgba(66,217,255,.15)" : "transparent", borderColor: forensicMode === m ? "rgba(66,217,255,.5)" : "rgba(151,204,227,.15)" }}>{m}</button>
              ))}
            </div>
            <div style={{ marginTop: 12, height: 90, border: "1px solid rgba(151,204,227,.15)", borderRadius: 6, display: "grid", placeItems: "center", color: "#6e909d", font: "8px IBM Plex Mono, monospace", background: "rgba(2,16,24,.6)" }}>
              {forensicMode} — {forensicMode === "HEATMAP" ? "Tamper heatmap overlay" : "Visualization"}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
