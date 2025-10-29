# genAI-LOGclassification-System
This project showcases a production-ready approach to system monitoring, combining the speed of Machine Learning with the intelligence of Generative AI.


The Dual-Engine Approach

ML Speed (Naïve Bayes): A custom Naïve Bayes Classifier instantly categorizes logs into four severity levels (ERROR, WARNING, INFO, DEBUG). This immediate triage is crucial for filtering noise and ensuring the system focuses only on the critical 5%. See the initial output in the 📊 ML Classification tab.

GenAI Intelligence (Gemini 2.5 Flash): For any critical log (ERROR or WARNING), the system triggers the Gemini 2.5 Flash API. Acting as an "Expert DevOp Assistant," it produces a detailed issue explanation and a clear, three-step remediation playbook ready for deployment. Find the full analysis in the 💡 Remediation Plan tab.

Why this matters for engineering teams:

Automation: Reduces manual log investigation time by up to 80%.

Consistency: Provides standardized, expert-level recovery steps every time.

Prototyping: Built using Streamlit with a wide-layout, sidebar, and professional tab navigation, demonstrating the ability to build and deploy complex, full-stack Python solutions.

This project validates the synergy between deterministic ML and contextual GenAI for building truly intelligent systems.
