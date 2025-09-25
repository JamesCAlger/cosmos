"""Base reporter interface and implementations for evaluation results"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import csv
from datetime import datetime
from loguru import logger


class BaseReporter(ABC):
    """Abstract base class for evaluation reporters"""

    @abstractmethod
    def report(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Generate and optionally save a report"""
        pass

    @abstractmethod
    def format_results(self, results: Dict[str, Any]) -> str:
        """Format results for display"""
        pass


class JSONReporter(BaseReporter):
    """JSON format reporter"""

    def __init__(self, indent: int = 2, include_metadata: bool = True):
        self.indent = indent
        self.include_metadata = include_metadata

    def report(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Generate JSON report"""
        # Add metadata if requested
        if self.include_metadata:
            results = self._add_metadata(results)

        # Format as JSON
        json_str = json.dumps(results, indent=self.indent, default=str)

        # Save if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(json_str)
            logger.info(f"JSON report saved to {output_path}")

        return json_str

    def format_results(self, results: Dict[str, Any]) -> str:
        """Format results as pretty JSON"""
        return json.dumps(results, indent=self.indent, default=str)

    def _add_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Add metadata to results"""
        return {
            "timestamp": datetime.now().isoformat(),
            "reporter": "JSONReporter",
            "results": results
        }


class CSVReporter(BaseReporter):
    """CSV format reporter for tabular data"""

    def __init__(self, flatten_nested: bool = True):
        self.flatten_nested = flatten_nested

    def report(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Generate CSV report"""
        # Flatten results for CSV
        rows = self._flatten_results(results) if self.flatten_nested else [results]

        # Create CSV string
        if rows:
            headers = list(rows[0].keys())
            csv_lines = [",".join(headers)]

            for row in rows:
                values = [str(row.get(h, "")) for h in headers]
                csv_lines.append(",".join(values))

            csv_str = "\n".join(csv_lines)

            # Save if path provided
            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(rows)
                logger.info(f"CSV report saved to {output_path}")

            return csv_str

        return ""

    def format_results(self, results: Dict[str, Any]) -> str:
        """Format results as CSV string"""
        return self.report(results)

    def _flatten_results(self, results: Dict[str, Any], parent_key: str = "") -> List[Dict]:
        """Flatten nested dictionary into list of flat dictionaries"""
        items = []

        # Handle list of results
        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict):
                    items.extend(self._flatten_results(item, parent_key))
            return items

        # Handle dictionary
        flat_dict = {}
        for k, v in results.items():
            new_key = f"{parent_key}.{k}" if parent_key else k

            if isinstance(v, dict):
                flat_dict.update(self._flatten_dict(v, new_key))
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                # List of dictionaries - create separate rows
                for i, item in enumerate(v):
                    row_dict = flat_dict.copy()
                    row_dict.update(self._flatten_dict(item, new_key))
                    items.append(row_dict)
            else:
                flat_dict[new_key] = v

        if not items:
            items = [flat_dict]

        return items

    def _flatten_dict(self, d: Dict, parent_key: str = "") -> Dict:
        """Flatten a single dictionary"""
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key))
            else:
                items[new_key] = v
        return items


class HTMLReporter(BaseReporter):
    """HTML format reporter with charts and tables"""

    def __init__(self, include_charts: bool = True, theme: str = "light"):
        self.include_charts = include_charts
        self.theme = theme

    def report(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Generate HTML report"""
        html = self._generate_html(results)

        # Save if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(html)
            logger.info(f"HTML report saved to {output_path}")

        return html

    def format_results(self, results: Dict[str, Any]) -> str:
        """Format results as HTML"""
        return self._generate_html(results)

    def _generate_html(self, results: Dict[str, Any]) -> str:
        """Generate complete HTML document"""
        # CSS styles
        css = self._get_css()

        # JavaScript for charts
        js = self._get_javascript() if self.include_charts else ""

        # Generate content
        content = self._generate_content(results)

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Evaluation Report</title>
    <style>{css}</style>
    {js}
</head>
<body class="{self.theme}">
    <div class="container">
        <h1>RAG Evaluation Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        {content}
    </div>
</body>
</html>
"""
        return html

    def _generate_content(self, results: Dict[str, Any]) -> str:
        """Generate HTML content from results"""
        sections = []

        # Summary section
        if "summary" in results:
            sections.append(self._generate_summary_section(results["summary"]))

        # Metrics section
        if "metrics" in results:
            sections.append(self._generate_metrics_section(results["metrics"]))

        # Statistical analysis section
        if "statistical_analysis" in results:
            sections.append(self._generate_stats_section(results["statistical_analysis"]))

        # Cost tracking section
        if "cost_summary" in results:
            sections.append(self._generate_cost_section(results["cost_summary"]))

        # Detailed results table
        if "detailed_results" in results:
            sections.append(self._generate_results_table(results["detailed_results"]))

        return "\n".join(sections)

    def _generate_summary_section(self, summary: Dict) -> str:
        """Generate summary section HTML"""
        items = []
        for key, value in summary.items():
            items.append(f"<li><strong>{key}:</strong> {value}</li>")

        return f"""
<section class="summary">
    <h2>Summary</h2>
    <ul>
        {' '.join(items)}
    </ul>
</section>
"""

    def _generate_metrics_section(self, metrics: Dict) -> str:
        """Generate metrics section with cards"""
        cards = []
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                cards.append(f"""
<div class="metric-card">
    <h3>{metric}</h3>
    <p class="metric-value">{value:.3f}</p>
</div>
""")

        return f"""
<section class="metrics">
    <h2>Metrics</h2>
    <div class="metric-grid">
        {' '.join(cards)}
    </div>
</section>
"""

    def _generate_stats_section(self, stats: Dict) -> str:
        """Generate statistical analysis section"""
        return f"""
<section class="statistics">
    <h2>Statistical Analysis</h2>
    <pre>{json.dumps(stats, indent=2)}</pre>
</section>
"""

    def _generate_cost_section(self, cost: Dict) -> str:
        """Generate cost tracking section"""
        return f"""
<section class="costs">
    <h2>Cost Tracking</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Cost</td><td>${cost.get('total_cost', 0):.4f}</td></tr>
        <tr><td>Input Tokens</td><td>{cost.get('total_input_tokens', 0):,}</td></tr>
        <tr><td>Output Tokens</td><td>{cost.get('total_output_tokens', 0):,}</td></tr>
    </table>
</section>
"""

    def _generate_results_table(self, results: List[Dict]) -> str:
        """Generate detailed results table"""
        if not results:
            return ""

        headers = list(results[0].keys())
        header_row = "".join(f"<th>{h}</th>" for h in headers)

        rows = []
        for result in results[:100]:  # Limit to 100 rows for performance
            row = "".join(f"<td>{result.get(h, '')}</td>" for h in headers)
            rows.append(f"<tr>{row}</tr>")

        return f"""
<section class="results">
    <h2>Detailed Results</h2>
    <table>
        <thead><tr>{header_row}</tr></thead>
        <tbody>{' '.join(rows)}</tbody>
    </table>
</section>
"""

    def _get_css(self) -> str:
        """Get CSS styles"""
        return """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background: #f5f5f5;
}

.container {
    background: white;
    padding: 30px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

h1 {
    color: #2c3e50;
    border-bottom: 3px solid #3498db;
    padding-bottom: 10px;
}

h2 {
    color: #34495e;
    margin-top: 30px;
    border-bottom: 1px solid #ecf0f1;
    padding-bottom: 5px;
}

.timestamp {
    color: #7f8c8d;
    font-size: 0.9em;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin: 20px 0;
}

.metric-card {
    background: #f8f9fa;
    padding: 20px;
    border-radius: 8px;
    border-left: 4px solid #3498db;
    text-align: center;
}

.metric-card h3 {
    margin: 0 0 10px 0;
    color: #555;
    font-size: 0.9em;
    text-transform: uppercase;
}

.metric-value {
    font-size: 2em;
    font-weight: bold;
    color: #2c3e50;
    margin: 0;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
}

th, td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid #ddd;
}

th {
    background: #34495e;
    color: white;
    font-weight: bold;
}

tr:hover {
    background: #f5f5f5;
}

pre {
    background: #f4f4f4;
    padding: 15px;
    border-radius: 4px;
    overflow-x: auto;
}

.dark {
    background: #1a1a1a;
    color: #e0e0e0;
}

.dark .container {
    background: #2a2a2a;
}

.dark h1, .dark h2 {
    color: #e0e0e0;
}

.dark table th {
    background: #3a3a3a;
}

.dark .metric-card {
    background: #3a3a3a;
    border-left-color: #5a9fd4;
}
"""

    def _get_javascript(self) -> str:
        """Get JavaScript for charts (placeholder for now)"""
        return """
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
// Chart initialization code would go here
</script>
"""


class MarkdownReporter(BaseReporter):
    """Markdown format reporter"""

    def __init__(self, include_toc: bool = True):
        self.include_toc = include_toc

    def report(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Generate Markdown report"""
        md = self._generate_markdown(results)

        # Save if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(md)
            logger.info(f"Markdown report saved to {output_path}")

        return md

    def format_results(self, results: Dict[str, Any]) -> str:
        """Format results as Markdown"""
        return self._generate_markdown(results)

    def _generate_markdown(self, results: Dict[str, Any]) -> str:
        """Generate Markdown document"""
        sections = []

        # Title and metadata
        sections.append("# RAG Evaluation Report")
        sections.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

        # Table of contents
        if self.include_toc:
            sections.append(self._generate_toc(results))

        # Sections
        if "summary" in results:
            sections.append(self._generate_summary_md(results["summary"]))

        if "metrics" in results:
            sections.append(self._generate_metrics_md(results["metrics"]))

        if "statistical_analysis" in results:
            sections.append(self._generate_stats_md(results["statistical_analysis"]))

        if "cost_summary" in results:
            sections.append(self._generate_cost_md(results["cost_summary"]))

        return "\n".join(sections)

    def _generate_toc(self, results: Dict) -> str:
        """Generate table of contents"""
        toc = ["## Table of Contents\n"]
        if "summary" in results:
            toc.append("- [Summary](#summary)")
        if "metrics" in results:
            toc.append("- [Metrics](#metrics)")
        if "statistical_analysis" in results:
            toc.append("- [Statistical Analysis](#statistical-analysis)")
        if "cost_summary" in results:
            toc.append("- [Cost Tracking](#cost-tracking)")
        return "\n".join(toc) + "\n"

    def _generate_summary_md(self, summary: Dict) -> str:
        """Generate summary section in Markdown"""
        lines = ["## Summary\n"]
        for key, value in summary.items():
            lines.append(f"- **{key}**: {value}")
        return "\n".join(lines) + "\n"

    def _generate_metrics_md(self, metrics: Dict) -> str:
        """Generate metrics section in Markdown"""
        lines = ["## Metrics\n"]
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")

        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                lines.append(f"| {metric} | {value:.3f} |")

        return "\n".join(lines) + "\n"

    def _generate_stats_md(self, stats: Dict) -> str:
        """Generate statistical analysis in Markdown"""
        lines = ["## Statistical Analysis\n"]
        lines.append("```json")
        lines.append(json.dumps(stats, indent=2))
        lines.append("```")
        return "\n".join(lines) + "\n"

    def _generate_cost_md(self, cost: Dict) -> str:
        """Generate cost tracking in Markdown"""
        lines = ["## Cost Tracking\n"]
        lines.append(f"- **Total Cost**: ${cost.get('total_cost', 0):.4f}")
        lines.append(f"- **Input Tokens**: {cost.get('total_input_tokens', 0):,}")
        lines.append(f"- **Output Tokens**: {cost.get('total_output_tokens', 0):,}")
        return "\n".join(lines) + "\n"


class CompositeReporter(BaseReporter):
    """Composite reporter that can generate multiple formats"""

    def __init__(self, reporters: Optional[List[BaseReporter]] = None):
        if reporters is None:
            # Default to all formats
            self.reporters = {
                "json": JSONReporter(),
                "csv": CSVReporter(),
                "html": HTMLReporter(),
                "markdown": MarkdownReporter()
            }
        else:
            self.reporters = {r.__class__.__name__.lower().replace("reporter", ""): r for r in reporters}

    def report(self, results: Dict[str, Any], output_path: Optional[str] = None) -> Dict[str, str]:
        """Generate reports in all configured formats"""
        reports = {}

        for format_name, reporter in self.reporters.items():
            # Generate output path for each format
            if output_path:
                base_path = Path(output_path).stem
                parent_dir = Path(output_path).parent
                format_path = parent_dir / f"{base_path}.{format_name}"
            else:
                format_path = None

            # Generate report
            reports[format_name] = reporter.report(results, str(format_path) if format_path else None)

        return reports

    def format_results(self, results: Dict[str, Any]) -> str:
        """Format results (uses first reporter)"""
        first_reporter = next(iter(self.reporters.values()))
        return first_reporter.format_results(results)