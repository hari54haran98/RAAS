"""
DAY 13: Logging + Monitoring for RAAS
Professional audit logging for every query
"""

import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
import os


class RAASLogger:
    """Professional logger for RAAS with audit trail."""

    def __init__(self, log_dir="logs"):
        print("📝 DAY 13: LOGGING + MONITORING")
        print("=" * 50)

        # Create logs directory
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Set up audit log (JSON format)
        self.audit_logger = logging.getLogger('raas_audit')
        self.audit_logger.setLevel(logging.INFO)

        # Rotating file handler (10 MB per file, keep 5 backups)
        audit_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'audit.json',
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        audit_handler.setFormatter(logging.Formatter('%(message)s'))
        self.audit_logger.addHandler(audit_handler)

        # Set up error log
        self.error_logger = logging.getLogger('raas_errors')
        self.error_logger.setLevel(logging.ERROR)

        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'errors.log',
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        error_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.error_logger.addHandler(error_handler)

        # Set up performance log
        self.perf_logger = logging.getLogger('raas_performance')
        self.perf_logger.setLevel(logging.INFO)

        perf_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'performance.csv',
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        perf_handler.setFormatter(logging.Formatter(
            '%(asctime)s,%(message)s'
        ))
        self.perf_logger.addHandler(perf_handler)

        # Write headers to performance CSV
        if not (self.log_dir / 'performance.csv').exists():
            self.perf_logger.info('timestamp,query,response_time_ms,confidence,hallucination_score,is_safe,num_sources')

        print(f"✓ Logs directory: {self.log_dir.absolute()}")
        print("✓ Audit log: audit.json")
        print("✓ Error log: errors.log")
        print("✓ Performance log: performance.csv")

        self.query_count = 0
        self.error_count = 0

    def log_query(self, question, answer, confidence, sources,
                  hallucination_score, is_safe, response_time_ms,
                  error=None, metadata=None):
        """
        Log a complete query with all details.

        Args:
            question: User question
            answer: Generated answer
            confidence: Confidence score
            sources: List of source documents
            hallucination_score: Hallucination detection score
            is_safe: Whether answer is safe
            response_time_ms: Response time in milliseconds
            error: Error message if any
            metadata: Additional metadata
        """
        timestamp = datetime.now().isoformat()
        self.query_count += 1

        # Create audit entry
        audit_entry = {
            'timestamp': timestamp,
            'query_id': f"Q{self.query_count:06d}",
            'question': question,
            'answer': answer,
            'confidence': confidence,
            'hallucination_score': hallucination_score,
            'is_safe': is_safe,
            'response_time_ms': response_time_ms,
            'num_sources': len(sources),
            'sources': sources,
            'error': error
        }

        if metadata:
            audit_entry['metadata'] = metadata

        # Write to audit log (JSON)
        self.audit_logger.info(json.dumps(audit_entry))

        # Write to performance log (CSV)
        self.perf_logger.info(
            f"{timestamp},{question[:50].replace(',', ' ')},"
            f"{response_time_ms},{confidence},{hallucination_score},"
            f"{is_safe},{len(sources)}"
        )

        # Log error separately if present
        if error:
            self.error_count += 1
            self.error_logger.error(
                f"Query {audit_entry['query_id']}: {error}\n"
                f"Question: {question}\n"
                f"Answer: {answer}"
            )

    def get_stats(self):
        """Get current logging statistics."""
        return {
            'total_queries': self.query_count,
            'total_errors': self.error_count,
            'log_dir': str(self.log_dir.absolute()),
            'log_files': {
                'audit': 'audit.json',
                'errors': 'errors.log',
                'performance': 'performance.csv'
            }
        }

    def generate_report(self):
        """Generate a summary report from logs."""
        print("\n📊 LOGGING SUMMARY REPORT")
        print("=" * 50)
        print(f"Total queries logged: {self.query_count}")
        print(f"Total errors logged: {self.error_count}")
        print(f"\nLog files location: {self.log_dir.absolute()}")
        print("\nFiles created:")
        print(f"  • audit.json - Complete JSON audit trail")
        print(f"  • errors.log - Error-only log")
        print(f"  • performance.csv - Performance metrics (CSV)")

        if self.query_count > 0:
            print("\n✅ Logging is active and working")
        else:
            print("\n⚠️ No queries logged yet")


# Quick test
if __name__ == "__main__":
    logger = RAASLogger()

    # Test log
    logger.log_query(
        question="What is the penalty?",
        answer="The penalty is 2.40% per annum.",
        confidence=0.95,
        sources=["sbi_home_loan_terms p2"],
        hallucination_score=0.15,
        is_safe=True,
        response_time_ms=4231.45
    )

    logger.generate_report()