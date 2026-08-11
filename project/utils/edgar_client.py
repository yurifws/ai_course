# Thin wrapper around edgartools for SEC filings.
# Fetches the latest 10-K / 10-Q for a ticker and extracts selected Item sections.

from typing import Any, Dict

from edgar import set_identity, Company


class EdgarClient:
    # Item numbers to pull per form type (Item 1A = risk factors, etc.).
    FORM_ITEMS = {"10-K": ["1", "1A", "7", "8", "9A"], "10-Q": ["1", "2", "3", "4"]}

    def __init__(self, email: str):
        # SEC requires a User-Agent identity (email) for EDGAR requests.
        set_identity(email)

    def fetch_filing_data(self, ticker: str, form_type: str) -> Dict[str, Any]:
        company = Company(ticker)
        # form= filters by filing type; latest() returns the most recent match.
        filing = company.get_filings(form=form_type).latest()

        metadata = {
            "ticker": ticker,
            "company_name": filing.company,
            "report_date": str(filing.report_date),
            "form_type": filing.form,
        }

        # filing.obj() exposes structured Item accessors (e.g. "Item 1A").
        filing_obj = filing.obj()
        items = {}

        for item_num in self.FORM_ITEMS[form_type]:
            item_key = f"Item {item_num}"
            try:
                items[item_key] = filing_obj[item_key]
            except (KeyError, IndexError):
                # Some filings omit sections; skip rather than fail the whole ingest.
                continue

        return {"metadata": metadata, "items": items}

    def get_combined_text(self, data: Dict) -> str:
        # Join selected items into one Markdown string for the semantic chunker.
        texts = []
        for item_name, item_content in data["items"].items():
            texts.append(f"## {item_name}\n\n{item_content}")

        return "\n\n".join(texts)
