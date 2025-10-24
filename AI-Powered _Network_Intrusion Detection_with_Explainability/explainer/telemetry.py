"""
RAG-Enhanced Performance Telemetry
Tracks latency, token usage, and tier distribution for hybrid explainer
"""
import time
import json
import logging
import numpy as np
from datetime import datetime
from collections import defaultdict
from pathlib import Path

class RAGTelemetry:
    """Production telemetry for RAG-based explainer system"""

    def __init__(self, log_file="logs/rag_telemetry.jsonl"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(exist_ok=True, parents=True)

        # Metrics storage
        self.metrics = defaultdict(list)
        self.session_start = datetime.now()

        # Configure logging
        self.logger = logging.getLogger("rag_telemetry")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self.log_file)
        handler.setFormatter(logging.Formatter("%(message)s"))
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def record_request(self, source, latency_ms, tokens_used=0, 
                      success=True, llm_used=False, mitre_techniques=None):
        """Record a single explanation request"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "latency_ms": round(latency_ms, 2),
            "tokens_used": tokens_used,
            "success": success,
            "llm_used": llm_used,
            "mitre_count": len(mitre_techniques) if mitre_techniques else 0
        }

        self.logger.info(json.dumps(record))
        self.metrics[source].append({
            "latency_ms": latency_ms,
            "tokens_used": tokens_used,
            "llm_used": llm_used
        })

    def get_summary(self):
        """Calculate comprehensive statistics"""
        summary = {}
        total_requests = 0
        total_tokens = 0
        llm_success_count = 0
        llm_fallback_count = 0

        for source, records in self.metrics.items():
            latencies = [r["latency_ms"] for r in records]
            tokens = [r["tokens_used"] for r in records]
            llm_used = [r.get("llm_used", False) for r in records]

            count = len(latencies)
            total_requests += count
            total_tokens += sum(tokens)

            if source == "rag_llm":
                llm_success_count += count
            elif source == "template_fallback":
                llm_fallback_count += count

            summary[source] = {
                "count": count,
                "latency_mean_ms": round(np.mean(latencies), 2) if latencies else 0,
                "latency_median_ms": round(np.median(latencies), 2) if latencies else 0,
                "latency_p95_ms": round(np.percentile(latencies, 95), 2) if latencies else 0,
                "latency_p99_ms": round(np.percentile(latencies, 99), 2) if latencies else 0,
                "tokens_total": sum(tokens),
                "tokens_mean": round(np.mean(tokens), 1) if tokens else 0
            }

        # Calculate tier distribution
        tier_percentages = {
            s: round(summary[s]["count"] / total_requests * 100, 1) 
            for s in summary
        } if total_requests > 0 else {}

        # Token savings calculation
        tokens_baseline = total_requests * 200
        tokens_saved = tokens_baseline - total_tokens
        savings_pct = round(tokens_saved / tokens_baseline * 100, 1) if tokens_baseline > 0 else 0

        # LLM success rate
        llm_total = llm_success_count + llm_fallback_count
        llm_success_rate = round(llm_success_count / llm_total * 100, 1) if llm_total > 0 else 0

        # Session duration
        session_duration = (datetime.now() - self.session_start).total_seconds()

        return {
            "summary": summary,
            "tier_distribution": tier_percentages,
            "total_requests": total_requests,
            "session_duration_sec": round(session_duration, 1),
            "requests_per_second": round(total_requests / session_duration, 2) if session_duration > 0 else 0,
            "token_savings": {
                "baseline_tokens": tokens_baseline,
                "actual_tokens": total_tokens,
                "tokens_saved": tokens_saved,
                "savings_percent": savings_pct
            },
            "llm_performance": {
                "success_count": llm_success_count,
                "fallback_count": llm_fallback_count,
                "success_rate_percent": llm_success_rate
            }
        }

    def print_summary(self):
        """Print human-readable summary"""
        stats = self.get_summary()

        print()
        print("=" * 70)
        print("RAG TELEMETRY SUMMARY")
        print("=" * 70)

        print()
        print("Overall Statistics:")
        print(f"   Total Requests: {stats['total_requests']}")
        print(f"   Session Duration: {stats['session_duration_sec']:.1f}s")
        print(f"   Throughput: {stats['requests_per_second']:.2f} req/s")

        print()
        print("Tier Distribution:")
        for tier, pct in sorted(stats['tier_distribution'].items(), key=lambda x: -x[1]):
            print(f"   {tier:20s}: {pct:5.1f}%")

        print()
        print("Latency by Tier:")
        for tier, metrics in stats['summary'].items():
            print(f"   {tier:20s}: {metrics['latency_mean_ms']:6.1f}ms (mean), "
                  f"{metrics['latency_p95_ms']:6.1f}ms (p95)")

        print()
        print("Token Savings:")
        print(f"   Baseline (no optimization): {stats['token_savings']['baseline_tokens']:,} tokens")
        print(f"   Actual usage: {stats['token_savings']['actual_tokens']:,} tokens")
        print(f"   Saved: {stats['token_savings']['tokens_saved']:,} tokens ({stats['token_savings']['savings_percent']}%)")

        print()
        print("LLM Performance:")
        print(f"   Successful generations: {stats['llm_performance']['success_count']}")
        print(f"   Template fallbacks: {stats['llm_performance']['fallback_count']}")
        print(f"   Success rate: {stats['llm_performance']['success_rate_percent']}%")

# Global telemetry instance
telemetry = RAGTelemetry()

def timed_explain(original_explain):
    """Decorator to track explanation performance"""
    def wrapper(self, features_dict, attack_type, prediction, confidence):
        start_time = time.time()
        try:
            result = original_explain(self, features_dict, attack_type, prediction, confidence)
            latency_ms = (time.time() - start_time) * 1000

            # Extract telemetry data
            source = result.get("source", "unknown")
            llm_used = result.get("llm_used", False)
            tokens = len(result.get("explanation", "")) // 4 if llm_used else 0
            mitre_techniques = result.get("mitre_techniques", [])

            telemetry.record_request(
                source=source,
                latency_ms=latency_ms,
                tokens_used=tokens,
                success=True,
                llm_used=llm_used,
                mitre_techniques=mitre_techniques
            )

            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            telemetry.record_request("error", latency_ms, 0, False)
            raise

    return wrapper
