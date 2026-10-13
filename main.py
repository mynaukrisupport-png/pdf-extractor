from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import fitz  # PyMuPDF
import requests
import tempfile
import os

app = FastAPI(
    title="PDF Recruitment Extractor",
    version="1.0.0"
)


class ExtractRequest(BaseModel):
    pdf_url: HttpUrl


@app.get("/")
def root():
    return {
        "success": True,
        "message": "PDF extractor is running"
    }


@app.post("/extract")
def extract_pdf(request: ExtractRequest):

    pdf_url = str(request.pdf_url)

    temp_file = None

    try:
        # -----------------------------
        # Download PDF
        # -----------------------------

        response = requests.get(
            pdf_url,
            timeout=60,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download PDF: HTTP {response.status_code}"
            )

        # Make temporary PDF file
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        )

        temp_file.write(response.content)
        temp_file.close()

        # -----------------------------
        # Open PDF
        # -----------------------------

        document = fitz.open(temp_file.name)

        full_text = []
        tables = []

        # -----------------------------
        # Extract each page
        # -----------------------------

        for page_number, page in enumerate(document, start=1):

            # -------------------------
            # Extract normal text
            # -------------------------

            text = page.get_text("text")

            full_text.append(
                f"\n\n===== PAGE {page_number} =====\n\n"
                + text
            )

            # -------------------------
            # Extract tables
            # -------------------------

            try:

                table_result = page.find_tables()

                for table_index, table in enumerate(
                    table_result.tables,
                    start=1
                ):

                    extracted_table = table.extract()

                    if not extracted_table:
                        continue

                    tables.append({
                        "page": page_number,
                        "table_number": table_index,
                        "data": extracted_table
                    })

            except Exception as table_error:

                # Table detection failure should
                # not stop normal text extraction.
                tables.append({
                    "page": page_number,
                    "error": str(table_error)
                })

        document.close()

        # -----------------------------
        # Final text
        # -----------------------------

        text = "".join(full_text)

        return {
            "success": True,

            "source": {
                "pdf_url": pdf_url
            },

            "pdf": {
                "pages": len(full_text),
                "characters": len(text),
                "words": len(text.split())
            },

            "text": text,

            "tables": tables
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:

        if temp_file and os.path.exists(temp_file.name):
            os.remove(temp_file.name)
