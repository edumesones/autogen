# Magentic-One QA System: Robust Screenshot & Multimodal Agent Demo

## Description

This project demonstrates an advanced AutoGen agent system for QA and automation, focused on robust handling of multimodal content (text + images) and taking **screenshots** of web pages. It solves the classic `'list' object has no attribute 'strip'` error in agents like MultimodalWebSurfer, enabling reliable navigation, capture, and visual analysis flows.

---

## Advantages of the Agent-Based Approach

- **Fixed WebSurfer**: The `FixedMultimodalWebSurfer` agent safely processes any multimodal response, preventing errors and enabling screenshots in complex QA flows.
- **Specialized Assistant**: An assistant agent guides screenshot capture, analyzes visual content, and coordinates interaction with WebSurfer.
- **Separation of Responsibilities**: Each agent has a clear role (navigation, capture, analysis), making extension and debugging easier.
- **Scalability**: The system can be easily extended with more agents or tools (e.g., image analysis, OCR, etc).
- **Robustness**: Safe handling of lists and multimodal objects prevents unexpected crashes and allows rich (text + image) responses.

---

## Requirements

- **Python 3.9+**
- **Node.js** and **npx** (for Playwright and AutoGen)
- **Playwright** (for browser automation and screenshots)
- **AutoGen v0.4+** and extensions
- See `requirements.txt` for Python dependencies

---

## Installation

1. Clone the repository and navigate to this directory:
   ```bash
   cd autogen_pruebas/screenshots
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Playwright and its browsers:
   ```bash
   playwright install
   ```
4. Set up your environment variables (`.env`):
   ```env
   OPENAI_API_KEY=your_openai_api_key
   MODEL=gpt-4o-mini
   # Optional: SCREENSHOTS_DIR=screenshots
   ```

---

## Usage

Run the main script:
```bash
python magentic_screenshots.py
```

- The system will show an example flow where an agent navigates to a page, takes a screenshot, and analyzes it.
- Screenshots are saved in the configured directory (default: `screenshots/`).
- The flow is robust to multimodal errors and can be extended for other visual QA cases.

---

## Example Advantage

- **Without the fix**: The WebSurfer agent may fail when receiving multimodal responses (lists of text + images), interrupting the QA flow.
- **With this system**: The fixed agent processes any response, enabling visual analysis, report generation, and advanced QA flows without crashes.

---

## Extension and Customization
- You can add more agents (OCR, image analysis, etc).
- The assistant can guide more complex flows (multi-page, visual data extraction, etc).
- The system is ideal for robustness testing, debugging, and automated visual QA.

---

## License
MIT License. See the repository for details. 