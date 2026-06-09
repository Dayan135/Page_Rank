import { BookOpen, Compass, Network } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

/** A centered, monospaced formula block (Fira Code via .font-mono). */
function Formula({ children }: { children: React.ReactNode }) {
  return (
    <div className="my-3 overflow-x-auto rounded-md border bg-muted/40 px-4 py-3 text-center font-mono text-sm tabular-nums">
      {children}
    </div>
  );
}

export default function LearnPage() {
  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div className="space-y-2">
        <h1 className="flex items-center gap-2 text-2xl font-semibold tracking-tight">
          <BookOpen className="h-6 w-6 text-primary" aria-hidden />
          Understanding PageRank
        </h1>
        <p className="text-muted-foreground">
          How this tool scores nodes — from the classic random surfer to the personalized,
          seed-relative variant it computes on the GPU.
        </p>
      </div>

      {/* PageRank */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Network className="h-5 w-5 text-primary" aria-hidden />
            PageRank — the random surfer
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm leading-relaxed">
          <p>
            Picture a surfer clicking links at random. PageRank is the long-run fraction of
            time they spend at each node: a node is important if many important nodes point to
            it. Importance flows along edges and concentrates on well-connected nodes.
          </p>
          <p>
            Encode the graph as a <strong>column-stochastic transition matrix</strong> <em>M</em>.
            Each column <em>j</em> spreads node <em>j</em>'s out-going probability over its
            targets (edge weights <em>w</em> optional; uniform if unweighted):
          </p>
          <Formula>
            M[i, j] = w(j → i) / Σ<sub>k</sub> w(j → k)
          </Formula>
          <p>
            At every step the surfer also <em>teleports</em> with probability 1 − α to a random
            node (α is the <strong>damping factor</strong>, typically 0.85). This keeps the walk
            from getting stuck and guarantees a unique solution. The rank vector <em>r</em>
            evolves as:
          </p>
          <Formula>
            r<sub>t+1</sub> = α · M · r<sub>t</sub> + (1 − α) · v
          </Formula>
          <p>
            For ordinary PageRank the teleport vector <em>v</em> is uniform,
            <span className="font-mono"> v = (1/N)·𝟙</span>. Iterating to a fixed point gives the
            stationary ranks:
          </p>
          <Formula>
            r = α · M · r + (1 − α) · v
          </Formula>
          <p className="text-muted-foreground">
            <strong>Dangling nodes</strong> (no out-edges) would leak probability mass; their
            column is replaced by a uniform <span className="font-mono">1/N</span> distribution so
            the matrix stays stochastic.
          </p>
        </CardContent>
      </Card>

      {/* Personalized PageRank */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Compass className="h-5 w-5 text-accent" aria-hidden />
            Personalized PageRank — relative to seeds
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm leading-relaxed">
          <p>
            Personalized PageRank (PPR) asks a sharper question: important{" "}
            <em>relative to whom?</em> Instead of teleporting to a random node, the surfer
            restarts only at a chosen set of <strong>seed nodes</strong> <em>S</em>. The math is
            identical — only the teleport vector changes:
          </p>
          <Formula>
            v = (1 / |S|) · Σ<sub>s ∈ S</sub> e<sub>s</sub>
            <span className="mx-2 text-muted-foreground">→</span>
            r = α · M · r + (1 − α) · v
          </Formula>
          <p>
            Here <span className="font-mono">e<sub>s</sub></span> is the indicator vector for seed{" "}
            <em>s</em>. The result measures proximity/relevance to the seeds: nodes near them
            score high, distant ones low. This powers “related items”, friend suggestions, and
            local community detection.
          </p>
          <p className="text-muted-foreground">
            Because PPR is <em>linear</em> in <em>v</em>, the seed-set score is just the average
            of the single-seed runs — which is exactly how the backend computes a batch of seeds
            at once.
          </p>
        </CardContent>
      </Card>

      {/* How this app computes it */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">How this tool computes it</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm leading-relaxed">
          <p>
            Your uploaded graph becomes a column-stochastic <em>M</em>, moved to the GPU in a
            custom <span className="font-mono">PBR</span> sparse format. The power iteration
            <Formula>
              X<sub>t+1</sub> = α · M · X<sub>t</sub> + (1 − α) · e<sub>s</sub>
            </Formula>
            runs as a sparse matrix–matrix multiply per step (one column per seed) until the
            update <span className="font-mono">‖X<sub>t+1</sub> − X<sub>t</sub>‖₁</span> drops
            below your <strong>tolerance</strong> or it hits <strong>max iterations</strong>.
          </p>
          <ul className="ml-5 list-disc space-y-1 text-muted-foreground">
            <li><strong>α (damping)</strong> — restart probability is 1 − α; higher α spreads influence further.</li>
            <li><strong>Max iterations / tolerance</strong> — when to stop the power iteration.</li>
            <li><strong>Seeds</strong> — the personalization set; <em>at least one is required</em>.</li>
            <li><strong>Top X</strong> — how many of the highest-ranked nodes to highlight.</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}
