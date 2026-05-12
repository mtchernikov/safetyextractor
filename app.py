import os
import json
import traceback
from typing import Any

import streamlit as st
from openai import OpenAI


def get_api_key() -> str | None:
    try:
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def pretty_json(obj: Any) -> str:
    try:
        if isinstance(obj, str):
            obj = json.loads(obj)
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


def try_parse_json(text: str) -> Any:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    if text.startswith("```"):
        cleaned = text.strip("`").replace("json\n", "", 1).replace("JSON\n", "", 1)
        try:
            return json.loads(cleaned)
        except Exception:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return {"raw_text": text}


def run_raw_prompt(api_key: str, model: str, prompt_template: str, user_text: str, allowed_relations: str, ontology_hint: str) -> dict:
    client = OpenAI(api_key=api_key)
    final_prompt = prompt_template.format(text=user_text, allowed_relations=allowed_relations, ontology_hint=ontology_hint)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a precise safety engineering extraction assistant. Return valid JSON only."},
            {"role": "user", "content": final_prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    return {"prompt_used": final_prompt, "raw_output": content, "parsed_output": try_parse_json(content)}


def get_dspy_module(api_key: str, model: str, use_optimizer: bool):
    import dspy
    os.environ["OPENAI_API_KEY"] = api_key
    lm = dspy.LM(f"openai/{model}")
    dspy.configure(lm=lm)

    class ExtractSafetyFacts(dspy.Signature):
        """Extract safety-relevant concepts, relations, unknowns and candidate inferences.
        Do not decide applicable standards. Use only allowed relation IDs. Output valid JSON strings.
        """
        text = dspy.InputField(desc="Free-text system or product description.")
        allowed_relations = dspy.InputField(desc="Comma-separated allowed relation IDs.")
        ontology_hint = dspy.InputField(desc="Comma-separated ontology concepts that should be preferred.")
        concepts_json = dspy.OutputField(desc='JSON array of canonical concept strings, e.g. ["oxygen", "plastic", "gas_path"].')
        relations_json = dspy.OutputField(desc='JSON array of [subject, relation, object] triples using only allowed relation IDs.')
        unknowns_json = dspy.OutputField(desc='JSON array of missing information questions or unknowns.')
        candidate_inferences_json = dspy.OutputField(desc='JSON array of objects with fields fact, basis, needs_validation, confidence.')

    module = dspy.Predict(ExtractSafetyFacts)
    if not use_optimizer:
        return module

    examples = [
        dspy.Example(
            text="The O2 line runs close to a relay. Plastic tubing is used in the same gas path.",
            allowed_relations="near, flows_through, located_inside, controls, contains",
            ontology_hint="oxygen, electrical_actuator, plastic, gas_path",
            concepts_json=json.dumps(["oxygen", "electrical_actuator", "plastic", "gas_path"]),
            relations_json=json.dumps([["oxygen", "flows_through", "gas_path"], ["relay", "near", "plastic_tubing"]]),
            unknowns_json=json.dumps(["oxygen concentration", "oxygen pressure", "relay ignition source relevance", "plastic material grade"]),
            candidate_inferences_json=json.dumps([{"fact": "relay may be ignition-relevant near oxygen/plastic context", "basis": "relay close to O2 line and plastic tubing", "needs_validation": True, "confidence": "medium"}]),
        ).with_inputs("text", "allowed_relations", "ontology_hint"),
        dspy.Example(
            text="The CGM sends glucose values to the insulin pump. The pump automatically adapts basal rate.",
            allowed_relations="sends_data_to, controls, affects, depends_on",
            ontology_hint="CGM, insulin_pump, glucose_data, dose_profile, closed_loop",
            concepts_json=json.dumps(["CGM", "insulin_pump", "glucose_data", "dose_profile", "closed_loop"]),
            relations_json=json.dumps([["CGM", "sends_data_to", "insulin_pump"], ["insulin_pump", "controls", "dose_profile"]]),
            unknowns_json=json.dumps(["fallback after CGM loss", "independent dose limits", "secure pairing"]),
            candidate_inferences_json=json.dumps([{"fact": "CGM data may influence insulin therapy", "basis": "pump automatically adapts basal rate from glucose values", "needs_validation": True, "confidence": "high"}]),
        ).with_inputs("text", "allowed_relations", "ontology_hint"),
    ]

    def parse_json_array(value):
        try:
            parsed = json.loads(str(value))
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    def extraction_metric(example, pred, trace=None):
        score = 0.0
        expected_concepts = set(parse_json_array(example.concepts_json))
        predicted_concepts = set(parse_json_array(pred.concepts_json))
        if expected_concepts:
            score += 0.45 * len(expected_concepts & predicted_concepts) / len(expected_concepts)
        allowed = {x.strip() for x in str(example.allowed_relations).split(",") if x.strip()}
        predicted_relations = parse_json_array(pred.relations_json)
        predicted_relation_ids = {r[1] for r in predicted_relations if isinstance(r, list) and len(r) >= 3}
        if predicted_relation_ids and predicted_relation_ids.issubset(allowed):
            score += 0.25
        expected_relations = {tuple(r) for r in parse_json_array(example.relations_json) if isinstance(r, list)}
        predicted_relations_set = {tuple(r) for r in predicted_relations if isinstance(r, list)}
        if expected_relations:
            score += 0.20 * len(expected_relations & predicted_relations_set) / len(expected_relations)
        if parse_json_array(pred.unknowns_json):
            score += 0.10
        return score

    try:
        optimizer = dspy.BootstrapFewShot(metric=extraction_metric, max_bootstrapped_demos=2, max_labeled_demos=2, max_rounds=1)
    except AttributeError:
        from dspy.teleprompt import BootstrapFewShot
        optimizer = BootstrapFewShot(metric=extraction_metric, max_bootstrapped_demos=2, max_labeled_demos=2, max_rounds=1)
    return optimizer.compile(module, trainset=examples)


def run_dspy(api_key: str, model: str, user_text: str, allowed_relations: str, ontology_hint: str, use_optimizer: bool) -> dict:
    module = get_dspy_module(api_key, model, use_optimizer)
    pred = module(text=user_text, allowed_relations=allowed_relations, ontology_hint=ontology_hint)
    return {
        "concepts": try_parse_json(getattr(pred, "concepts_json", "")),
        "relations": try_parse_json(getattr(pred, "relations_json", "")),
        "unknowns": try_parse_json(getattr(pred, "unknowns_json", "")),
        "candidate_inferences": try_parse_json(getattr(pred, "candidate_inferences_json", "")),
        "raw_fields": {
            "concepts_json": getattr(pred, "concepts_json", ""),
            "relations_json": getattr(pred, "relations_json", ""),
            "unknowns_json": getattr(pred, "unknowns_json", ""),
            "candidate_inferences_json": getattr(pred, "candidate_inferences_json", ""),
        },
    }


st.set_page_config(page_title="Prompt vs DSPy Extraction Demo", layout="wide")
st.title("Prompt vs DSPy — Safety Co-Pilot Extraction Demo")
st.markdown("""
This app compares two ways to do the same extraction task:

- **Left:** a normal hand-written prompt sent directly to OpenAI.
- **Right:** a DSPy signature/module. Optionally, DSPy can compile a small few-shot extractor from examples.

The purpose is only to compare extraction behavior. Hazard templates and norm ranking are intentionally not included here.
""")

api_key = get_api_key()
if not api_key:
    st.error("No OPENAI_API_KEY found. Add it to Streamlit secrets or your environment.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    model = st.text_input("Model", value="gpt-5.5")
    use_optimizer = st.checkbox("Use DSPy BootstrapFewShot optimizer", value=False, help="Costs extra LLM calls on first run. Leave off for a quick comparison.")
    st.caption("For Streamlit Cloud, set OPENAI_API_KEY in app secrets.")

default_text = """The product is a lung ventilator.
The pressure profile is software-controlled.
Oxygen flow is regulated by an electrically controlled valve.
Plastic components are used in the gas path."""

default_relations = "near, located_inside, contains, controls, flows_through, sends_data_to, affects, part_of"
default_ontology = "medical_ventilator, oxygen, electrical_actuator, plastic, gas_path, therapy_profile, software_control, CGM, insulin_pump, dose_profile"

default_prompt = """Extract safety-relevant structured facts from the system description.

Rules:
- Do not decide applicable standards.
- Do not produce final hazard conclusions.
- Use only the allowed relation IDs.
- Return valid JSON only.
- Separate explicit facts from candidate inferences.

Allowed relations:
{allowed_relations}

Ontology hint:
{ontology_hint}

System description:
{text}

Return this JSON object:
{{
  "concepts": [],
  "relations": [
    ["subject", "relation", "object"]
  ],
  "unknowns": [],
  "candidate_inferences": [
    {{
      "fact": "",
      "basis": "",
      "needs_validation": true,
      "confidence": "low|medium|high"
    }}
  ]
}}
"""

st.subheader("Input")
user_text = st.text_area("System description", value=default_text, height=160)
allowed_relations = st.text_area("Allowed relations", value=default_relations, height=70)
ontology_hint = st.text_area("Ontology hint", value=default_ontology, height=70)

with st.expander("Normal prompt template"):
    prompt_template = st.text_area("Prompt template", value=default_prompt, height=360, help="Available placeholders: {text}, {allowed_relations}, {ontology_hint}")

if st.button("Run comparison", type="primary"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Without DSPy: raw prompt")
        try:
            raw_result = run_raw_prompt(api_key, model, prompt_template, user_text, allowed_relations, ontology_hint)
            st.markdown("**Parsed output**")
            st.code(pretty_json(raw_result["parsed_output"]), language="json")
            with st.expander("Raw model output"):
                st.code(raw_result["raw_output"])
            with st.expander("Prompt actually sent"):
                st.text(raw_result["prompt_used"])
        except Exception as e:
            st.error("Raw prompt call failed.")
            st.code(str(e))
            st.code(traceback.format_exc())
    with col2:
        st.subheader("With DSPy: signature/module")
        try:
            dspy_result = run_dspy(api_key, model, user_text, allowed_relations, ontology_hint, use_optimizer)
            st.markdown("**Structured output**")
            st.code(pretty_json({
                "concepts": dspy_result["concepts"],
                "relations": dspy_result["relations"],
                "unknowns": dspy_result["unknowns"],
                "candidate_inferences": dspy_result["candidate_inferences"],
            }), language="json")
            with st.expander("Raw DSPy output fields"):
                st.code(pretty_json(dspy_result["raw_fields"]), language="json")
            with st.expander("DSPy signature used"):
                st.code("""class ExtractSafetyFacts(dspy.Signature):
    text = dspy.InputField()
    allowed_relations = dspy.InputField()
    ontology_hint = dspy.InputField()

    concepts_json = dspy.OutputField()
    relations_json = dspy.OutputField()
    unknowns_json = dspy.OutputField()
    candidate_inferences_json = dspy.OutputField()
""", language="python")
        except Exception as e:
            st.error("DSPy call failed.")
            st.code(str(e))
            st.code(traceback.format_exc())

st.divider()
st.markdown("""
### What to look for

For the default ventilator example, a better extraction should identify:

```text
medical_ventilator
therapy_profile
software_control
oxygen
electrical_actuator
plastic
gas_path
```

and relations like:

```text
software_control controls therapy_profile
electrical_actuator controls oxygen_flow
plastic_component located_inside gas_path
oxygen flows_through gas_path
```

In a full Safety Co-Pilot, this extraction would then go into:

```text
ontology normalization → graph → propagation rules → hazard templates → combination templates → norm ranking
```
""")
