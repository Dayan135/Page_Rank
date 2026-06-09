import { Routes, Route, Navigate } from "react-router-dom";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/toaster";
import { AppShell } from "@/components/layout/AppShell";
import UploadPage from "@/pages/UploadPage";
import ConfigurePage from "@/pages/ConfigurePage";
import ResultsPage from "@/pages/ResultsPage";
import LearnPage from "@/pages/LearnPage";

export default function App() {
  return (
    <TooltipProvider delayDuration={300}>
      <AppShell>
        <Routes>
          <Route path="/" element={<Navigate to="/upload" replace />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/configure" element={<ConfigurePage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/learn" element={<LearnPage />} />
          <Route path="*" element={<Navigate to="/upload" replace />} />
        </Routes>
      </AppShell>
      <Toaster />
    </TooltipProvider>
  );
}
