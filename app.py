import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import yaml
from openai import OpenAI


def get_api_key() -> Optional[str]:
    try:
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def load_yaml_from_upload_or_default(uploaded_file, default_path: str) -> Dict[str, Any]:
    if uploaded_file is not None:
        return yaml.safe_load(uploaded_file.read().decode("utf-8"))
    with open(default_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def as_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def try_parse_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
        try:
            return json.loads(text)
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


def normalize_id(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def graph_has_concept(graph: List[Dict[str, Any]], concept: str) -> bool:
    concept = normalize_id(concept)
    for f in graph:
        if normalize_id(f.get("subject")) == concept or normalize_id(f.get("object")) == concept:
            return True
    return False


def graph_has_any_concept(graph: List[Dict[str, Any]], concepts: List[str]) -> bool:
    return any(graph_has_concept(graph, c) for c in concepts)


def add_fact(graph: List[Dict[str, Any]], subject: str, relation: str, object_: str, relation_group: str = "unknown", source: str = "derived", confidence: float = 0.7, evidence: str = "", line_id: Optional[int] = None, rule: Optional[str] = None, needs_validation: bool = False) -> None:
    fact = {
        "subject": normalize_id(subject),
        "relation": relation,
        "object": normalize_id(object_),
        "relation_group": relation_group,
        "source": source,
        "confidence": confidence,
        "evidence": evidence,
        "line_id": line_id,
        "needs_validation": needs_validation,
    }
    if rule:
        fact["rule"] = rule
    key = (fact["subject"], fact["relation"], fact["object"], fact["source"], fact.get("rule"))
    existing = {(x.get("subject"), x.get("relation"), x.get("object"), x.get("source"), x.get("rule")) for x in graph}
    if key not in existing:
        graph.append(fact)


def ontology_summary(ontology: Dict[str, Any]) -> str:
    lines = []
    for cid, c in ontology.get("concepts", {}).items():
        syns = ", ".join(c.get("synonyms", [])[:8])
        props = ", ".join(c.get("properties", [])[:8])
        lines.append(f"- {cid}: type={c.get('type')}; synonyms=[{syns}]; properties=[{props}]")
    return "\n".join(lines)


def allowed_relation_summary(relations: Dict[str, Any]) -> str:
    lines = []
    for rid, r in relations.get("relations", {}).items():
        lines.append(f"- {rid}: {r.get('description', '')}")
    return "\n".join(lines)


def llm_normalize_description(api_key: str, model: str, description: str, ontology: Dict[str, Any], relations: Dict[str, Any]) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    prompt = f"""
You are a safety engineering extraction and normalization module.

Task:
Normalize the system description into controlled concepts and graph triples.

Rules:
- Do NOT decide final hazards or applicable standards.
- Use the ontology concepts where possible.
- Use only allowed relation IDs.
- Mark inferred facts as source="inferred".
- Mark direct facts from the text as source="explicit".
- Return valid JSON only.

Ontology:
{ontology_summary(ontology)}

Allowed relations:
{allowed_relation_summary(relations)}

System description:
{description}

Return JSON:
{{
  "entities": [
    {{
      "id": "E1",
      "original_text": "",
      "canonical": "",
      "type": "",
      "line_id": 1,
      "confidence": "low|medium|high"
    }}
  ],
  "triples": [
    {{
      "subject": "",
      "relation": "",
      "object": "",
      "relation_group": "",
      "source": "explicit|inferred",
      "line_id": 1,
      "confidence": 0.0,
      "evidence": ""
    }}
  ],
  "unknowns": []
}}
"""
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return valid JSON only. Be conservative and traceable."},
            {"role": "user", "content": prompt},
        ],
    )
    return try_parse_json(response.choices[0].message.content or "{}")


def fallback_normalize(description: str, ontology: Dict[str, Any]) -> Dict[str, Any]:
    text = description.lower()
    entities, triples = [], []
    def ent(original, canonical, typ):
        entities.append({"id": f"E{len(entities)+1}", "original_text": original, "canonical": canonical, "type": typ, "line_id": 1, "confidence": "medium"})
    if "ventilator" in text or "respirator" in text:
        ent("ventilator", "medical_ventilator", "device")
    if "pressure" in text:
        ent("pressure", "airway_pressure", "controlled_parameter")
    if "lung" in text:
        ent("lung", "lung", "body_part")
    if "patient" in text:
        ent("patient", "patient", "person")
    if "plastic" in text and ("tube" in text or "tubing" in text):
        ent("plastic tube", "plastic_tube", "component")
    if "oxygen" in text or "o2" in text:
        ent("oxygen", "oxygen", "oxidizer")
    if "valve" in text or "relay" in text:
        ent("electronic valve module", "electrical_actuator", "component")
    concepts = {e["canonical"] for e in entities}
    if "medical_ventilator" in concepts and "airway_pressure" in concepts:
        triples.append({"subject": "medical_ventilator", "relation": "controls", "object": "airway_pressure", "relation_group": "control", "source": "explicit", "line_id": 1, "confidence": 0.8, "evidence": "ventilator applies pressure"})
    if "airway_pressure" in concepts and "plastic_tube" in concepts:
        triples.append({"subject": "airway_pressure", "relation": "transmitted_through", "object": "plastic_tube", "relation_group": "medium_flow", "source": "explicit", "line_id": 1, "confidence": 0.8, "evidence": "via a plastic tube"})
    if "plastic_tube" in concepts:
        triples.append({"subject": "plastic_tube", "relation": "material", "object": "plastic", "relation_group": "material", "source": "explicit", "line_id": 1, "confidence": 0.8, "evidence": "plastic tube"})
    if "plastic_tube" in concepts and "patient" in concepts:
        triples.append({"subject": "plastic_tube", "relation": "connected_to", "object": "patient_airway", "relation_group": "structural", "source": "inferred", "line_id": 1, "confidence": 0.65, "evidence": "tube connects ventilator therapy to patient"})
    if "patient" in concepts and "lung" in concepts:
        triples.append({"subject": "patient_airway", "relation": "connected_to", "object": "lung", "relation_group": "physiological", "source": "inferred", "line_id": 1, "confidence": 0.65, "evidence": "patient lung"})
    if "oxygen" in concepts and "plastic_tube" in concepts:
        triples.append({"subject": "oxygen", "relation": "flows_through", "object": "plastic_tube", "relation_group": "medium_flow", "source": "inferred", "line_id": 1, "confidence": 0.6, "evidence": "oxygen context and plastic tube"})
    if "electrical_actuator" in concepts and "plastic_tube" in concepts:
        triples.append({"subject": "electrical_actuator", "relation": "near", "object": "plastic_tube", "relation_group": "structural", "source": "inferred", "line_id": 1, "confidence": 0.5, "evidence": "electronic valve module near tube"})
    return {"entities": entities, "triples": triples, "unknowns": ["patient group", "pressure limits", "tube material grade", "single-use or reusable tube", "patient-side filter", "oxygen concentration if oxygen is used"]}


def build_graph(normalized: Dict[str, Any], ontology: Dict[str, Any]) -> List[Dict[str, Any]]:
    graph: List[Dict[str, Any]] = []
    concepts = ontology.get("concepts", {})
    for ent in normalized.get("entities", []):
        canonical = normalize_id(ent.get("canonical") or ent.get("original_text"))
        typ = normalize_id(ent.get("type", "unknown"))
        add_fact(graph, canonical, "is_a", typ, "classification", "llm_extraction", {"high": 0.9, "medium": 0.7, "low": 0.4}.get(ent.get("confidence"), 0.7), ent.get("original_text", ""), ent.get("line_id"))
    for tr in normalized.get("triples", []):
        add_fact(graph, tr.get("subject", ""), tr.get("relation", "related_to"), tr.get("object", ""), tr.get("relation_group", "unknown"), tr.get("source", "explicit"), float(tr.get("confidence", 0.7) or 0.7), tr.get("evidence", ""), tr.get("line_id"))
    changed = True
    while changed:
        changed = False
        all_nodes = {f["subject"] for f in graph} | {f["object"] for f in graph}
        for cid, c in concepts.items():
            cid_norm = normalize_id(cid)
            if cid_norm not in all_nodes:
                continue
            before = len(graph)
            for prop in c.get("properties", []):
                add_fact(graph, cid_norm, "has_property", prop, "ontology_property", "ontology", 0.75, f"Ontology property of {cid}")
            for implied in c.get("implied_facts", []):
                add_fact(graph, implied.get("subject", cid_norm).replace("$self", cid_norm), implied.get("relation", "related_to"), implied.get("object", ""), implied.get("relation_group", "ontology_implied"), "ontology", float(implied.get("confidence", 0.7)), f"Ontology implied fact of {cid}", needs_validation=bool(implied.get("needs_validation", False)))
            if len(graph) > before:
                changed = True
    return graph


def has_fact(graph: List[Dict[str, Any]], subject: Optional[str] = None, relation: Optional[str] = None, object_: Optional[str] = None) -> bool:
    for f in graph:
        if subject and normalize_id(f.get("subject")) != normalize_id(subject):
            continue
        if relation and f.get("relation") != relation:
            continue
        if object_ and normalize_id(f.get("object")) != normalize_id(object_):
            continue
        return True
    return False


def apply_propagation_rules(graph: List[Dict[str, Any]], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    before = list(graph)
    if graph_has_any_concept(graph, ["medical_ventilator"]) and graph_has_any_concept(graph, ["airway_pressure", "therapy_profile", "pressure_profile"]):
        add_fact(graph, "airway_pressure", "may_cause_if_wrong", "lung_injury", "causal_hazard", "propagation_rule", 0.8, "Ventilator pressure/therapy profile affects lung mechanics.", rule="PR_THERAPY_PRESSURE_LUNG_HARM", needs_validation=True)
    tube_to_patient = has_fact(graph, "plastic_tube", "connected_to", "patient_airway") or has_fact(graph, "plastic_tube", "connected_to", "lung") or has_fact(graph, "airway_pressure", "transmitted_through", "plastic_tube")
    if tube_to_patient:
        add_fact(graph, "plastic_tube", "part_of", "patient_gas_path", "structural", "propagation_rule", 0.75, "Tube transmits therapy toward patient airway/lung.", rule="PR_PATIENT_GAS_PATH_FROM_TUBE", needs_validation=True)
        add_fact(graph, "patient_gas_path", "connected_to", "patient_airway", "structural", "propagation_rule", 0.7, "Patient gas path is connected to patient airway.", rule="PR_PATIENT_GAS_PATH_FROM_TUBE", needs_validation=True)
        add_fact(graph, "patient_airway", "connected_to", "lung", "physiological", "propagation_rule", 0.7, "Patient airway connects to lung.", rule="PR_PATIENT_GAS_PATH_FROM_TUBE", needs_validation=True)
    if has_fact(graph, "patient_gas_path", "connected_to", "patient_airway"):
        add_fact(graph, "patient_gas_path", "may_transport_if_present", "foreign_particle", "potential_medium_flow", "propagation_rule", 0.65, "A patient-connected gas path can transport particles/residues if present.", rule="PR_PATIENT_GAS_PATH_TRANSPORT_SUSCEPTIBILITY", needs_validation=True)
        add_fact(graph, "foreign_particle", "may_reach", "lung", "potential_medium_flow", "propagation_rule", 0.55, "Particle in patient gas path may reach patient airway/lung if not filtered.", rule="PR_PATIENT_GAS_PATH_TRANSPORT_SUSCEPTIBILITY", needs_validation=True)
    oxygen_present = graph_has_any_concept(graph, ["oxygen", "oxidizer"])
    fuel_present = graph_has_any_concept(graph, ["plastic", "combustible_material", "possible_fuel"])
    ignition_present = graph_has_any_concept(graph, ["electrical_actuator", "possible_ignition_source", "relay", "hot_surface", "spark", "arc"])
    locality = has_fact(graph, "electrical_actuator", "near", "plastic_tube") or has_fact(graph, "oxygen", "flows_through", "plastic_tube") or has_fact(graph, "oxygen", "flows_through", "patient_gas_path") or has_fact(graph, "plastic_tube", "part_of", "patient_gas_path")
    if oxygen_present and fuel_present and ignition_present and locality:
        add_fact(graph, "system", "has_potential_hazard_pattern", "oxygen_fire_triangle", "causal_hazard", "propagation_rule", 0.65, "Oxygen, possible fuel and possible ignition source appear in a related local context.", rule="PR_FIRE_TRIANGLE_POTENTIAL", needs_validation=True)
    return [f for f in graph if f not in before]


def trigger_item_matches(graph: List[Dict[str, Any]], item: Dict[str, Any]) -> Tuple[bool, List[str]]:
    evidence = []
    if "concepts_any" in item:
        for c in item["concepts_any"]:
            if graph_has_concept(graph, c):
                return True, [f"concept present: {c}"]
        return False, []
    if "relation" in item:
        spec = item["relation"]
        sub_any = [normalize_id(x) for x in spec.get("subject_any", [])]
        rel_any = spec.get("relation_any", [])
        obj_any = [normalize_id(x) for x in spec.get("object_any", [])]
        for f in graph:
            if (not sub_any or normalize_id(f.get("subject")) in sub_any) and (not rel_any or f.get("relation") in rel_any) and (not obj_any or normalize_id(f.get("object")) in obj_any):
                return True, [f'{f.get("subject")} — {f.get("relation")} → {f.get("object")}']
        return False, []
    if "relation_optional" in item:
        _, ev = trigger_item_matches(graph, {"relation": item["relation_optional"]})
        return True, ev
    return False, evidence


def match_hazard_templates(graph: List[Dict[str, Any]], hazard_templates: Dict[str, Any]) -> List[Dict[str, Any]]:
    matches = []
    for template in hazard_templates.get("templates", []):
        all_ok, evidence = True, []
        for item in template.get("trigger", {}).get("all_of", []):
            ok, ev = trigger_item_matches(graph, item)
            if not ok:
                all_ok = False
                break
            evidence.extend(ev)
        if all_ok:
            matches.append({
                "hazard": template.get("title", template.get("id")),
                "hazard_id": template.get("id"),
                "source": "Deterministic forward",
                "rationale": template.get("reasoning", "") + "\nEvidence: " + "; ".join(evidence),
                "confidence": template.get("confidence_default", "medium"),
                "missing_information": "; ".join(template.get("missing_information", [])),
            })
    return matches


def llm_backward_investigation(api_key: str, model: str, description: str, graph: List[Dict[str, Any]], hazard_templates: Dict[str, Any]) -> List[Dict[str, Any]]:
    client = OpenAI(api_key=api_key)
    compact_templates = [{"id": t.get("id"), "title": t.get("title"), "reasoning": t.get("reasoning"), "hazards": t.get("hazards", []), "trigger": t.get("trigger", {}), "missing_information": t.get("missing_information", [])} for t in hazard_templates.get("templates", [])]
    compact_graph = [{"subject": f.get("subject"), "relation": f.get("relation"), "object": f.get("object"), "source": f.get("source"), "confidence": f.get("confidence"), "needs_validation": f.get("needs_validation"), "evidence": f.get("evidence")} for f in graph]
    prompt = f"""
You are doing backward safety investigation.

Task:
For each hazard template, start from the hazard and investigate whether the design graph contains ingredients that make this hazard possible.

Rules:
- Use the system graph as the main evidence.
- Use the original description only as supporting evidence.
- Do not invent facts.
- Do not claim a hazard is confirmed unless the graph explicitly supports it.
- Classify as: confirmed_context, strong_potential, potential_pathway, partial_evidence, no_evidence.
- Be conservative.
- Return valid JSON only.

Original system description:
{description}

System graph facts:
{as_json(compact_graph)}

Hazard templates:
{as_json(compact_templates)}

Return JSON:
{{
  "results": [
    {{
      "hazard_id": "",
      "hazard": "",
      "status": "confirmed_context|strong_potential|potential_pathway|partial_evidence|no_evidence",
      "rationale": "",
      "confidence": "low|medium|high",
      "missing_information": []
    }}
  ]
}}
"""
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return valid JSON only. Investigate hazard templates against the graph."},
            {"role": "user", "content": prompt},
        ],
    )
    parsed = try_parse_json(response.choices[0].message.content or "{}")
    rows = []
    for r in parsed.get("results", []):
        if r.get("status") == "no_evidence":
            continue
        rows.append({
            "hazard": r.get("hazard", r.get("hazard_id", "")),
            "hazard_id": r.get("hazard_id", ""),
            "source": "LLM backward",
            "rationale": f"Status: {r.get('status')}. {r.get('rationale', '')}",
            "confidence": r.get("confidence", "medium"),
            "missing_information": "; ".join(r.get("missing_information", [])),
        })
    return rows


def graph_to_dot(graph: List[Dict[str, Any]]) -> str:
    def node_id(x: str) -> str:
        return "n_" + normalize_id(x)
    group_colors = {"classification": "gray", "structural": "blue", "functional": "green", "control": "orange", "medium_flow": "purple", "potential_medium_flow": "purple", "material": "brown", "physiological": "red", "causal_hazard": "red", "ontology_property": "gray", "hazard_role": "red", "ontology_implied": "gray"}
    lines = ["digraph G {", '  graph [rankdir=LR, bgcolor="transparent"];', '  node [shape=box, style="rounded,filled", fillcolor="white", color="#444444", fontname="Arial"];', '  edge [fontname="Arial", color="#555555"];']
    nodes = sorted({f.get("subject") for f in graph} | {f.get("object") for f in graph})
    for n in nodes:
        if n:
            lines.append(f'  {node_id(n)} [label="{str(n).replace("_", " ")}"];')
    for f in graph:
        s, o, r = f.get("subject"), f.get("object"), f.get("relation")
        if not s or not o:
            continue
        color = group_colors.get(f.get("relation_group"), "black")
        style = "dashed" if f.get("needs_validation") else "solid"
        lines.append(f'  {node_id(s)} -> {node_id(o)} [label="{str(r).replace(chr(34), chr(39))}", color="{color}", style="{style}"];')
    lines.append("}")
    return "\n".join(lines)


st.set_page_config(page_title="Safety Co-Pilot Ventilator PoC", layout="wide")
st.title("Safety Co-Pilot PoC — Lung Ventilator")
st.markdown("""
This proof of concept uses uploaded YAML knowledge files, then performs:

```text
system description → LLM normalization → Safety Context Graph → deterministic forward hazard matching → LLM backward hazard investigation → hazard table
```
""")
api_key = get_api_key()
with st.sidebar:
    st.header("Configuration")
    model = st.text_input("OpenAI model", value="gpt-4o-mini")
    st.caption("OPENAI_API_KEY is read from Streamlit Secrets or environment.")
    st.header("Upload YAML artifacts")
    ontology_upload = st.file_uploader("ontology.yaml", type=["yaml", "yml"])
    relations_upload = st.file_uploader("relations.yaml", type=["yaml", "yml"])
    propagation_upload = st.file_uploader("propagation_rules.yaml", type=["yaml", "yml"])
    hazard_upload = st.file_uploader("hazard_templates.yaml", type=["yaml", "yml"])
    use_fallback_if_no_key = st.checkbox("Use fallback extractor if no API key", value=True)

ontology = load_yaml_from_upload_or_default(ontology_upload, "data/ontology.yaml")
relations = load_yaml_from_upload_or_default(relations_upload, "data/relations.yaml")
propagation_rules = load_yaml_from_upload_or_default(propagation_upload, "data/propagation_rules.yaml")
hazard_templates = load_yaml_from_upload_or_default(hazard_upload, "data/hazard_templates.yaml")

default_description = """The product is a lung ventilator.
The ventilator applies pressure to the patient's lung via a plastic tube.
The oxygen concentration can be adjusted.
The plastic tube passes near an electronic valve module."""
st.subheader("1. System description")
description = st.text_area("Input or edit the system description. Press Proceed when ready.", value=default_description, height=170)
proceed = st.button("Proceed", type="primary")
if proceed:
    if not api_key and not use_fallback_if_no_key:
        st.error("No OPENAI_API_KEY found. Add it to Streamlit Secrets or enable fallback extractor.")
        st.stop()
    with st.spinner("Normalizing description and building graph..."):
        if api_key:
            normalized = llm_normalize_description(api_key, model, description, ontology, relations)
            normalization_source = "LLM"
        else:
            normalized = fallback_normalize(description, ontology)
            normalization_source = "Fallback extractor"
        graph = build_graph(normalized, ontology)
        derived = apply_propagation_rules(graph, propagation_rules)
        forward_rows = match_hazard_templates(graph, hazard_templates)
        backward_rows = llm_backward_investigation(api_key, model, description, graph, hazard_templates) if api_key else []
        all_rows = forward_rows + backward_rows
    st.success(f"Analysis complete. Normalization source: {normalization_source}")
    st.subheader("2. LLM-normalized description")
    st.json(normalized)
    st.subheader("3. Safety Context Graph")
    graph_tab, triples_tab, derived_tab = st.tabs(["Graph view", "Graph triples", "Derived facts"])
    with graph_tab:
        try:
            st.graphviz_chart(graph_to_dot(graph), use_container_width=True)
        except Exception as e:
            st.warning(f"Graph rendering failed: {e}")
            st.code(graph_to_dot(graph))
    with triples_tab:
        st.dataframe(pd.DataFrame(graph), use_container_width=True, hide_index=True)
    with derived_tab:
        if derived:
            st.dataframe(pd.DataFrame(derived), use_container_width=True, hide_index=True)
        else:
            st.info("No derived facts were added.")
    st.subheader("4. Forward deterministic + backward LLM hazard analysis")
    if all_rows:
        result_df = pd.DataFrame(all_rows).rename(columns={"hazard": "Hazard", "source": "LLM or Deterministic", "rationale": "Rationale why this hazard is possible", "confidence": "Confidence level", "missing_information": "Missing information"})
        result_df = result_df[["Hazard", "LLM or Deterministic", "Rationale why this hazard is possible", "Confidence level", "Missing information"]]
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        st.download_button("Download hazard table as CSV", data=result_df.to_csv(index=False).encode("utf-8"), file_name="safety_copilot_hazard_table.csv", mime="text/csv")
    else:
        st.info("No hazards or potential pathways were found.")
else:
    st.info("Upload YAML files if you want to override the defaults, then edit the description and press Proceed.")
