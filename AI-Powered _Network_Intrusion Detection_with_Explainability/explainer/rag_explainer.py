"""
Production RAG Explainer for MITRE ATT&CK
Integrates semantic search with Llama3.1:8b LLM for natural language explanations
"""

import json
import numpy as np


class ProductionRAGExplainer:
    """
    Production-grade RAG explainer with:
    - Semantic search using embeddings (ChromaDB)
    - Keyword-based fallback mapping
    - Llama3.1:8b LLM for natural language generation
    - Deterministic MITRE technique IDs (from retrieval only)
    - Flow feature integration in explanations
    - Zero LLM hallucinations on technique IDs
    """

    def __init__(self, mitre_kb_path, collection):
        # Load MITRE knowledge base
        with open(mitre_kb_path, 'r', encoding='utf-8') as f:
            self.mitre_kb = json.load(f)

        self.collection = collection

        # Strict attack type to technique mapping (47 techniques)
        self.technique_mapping = {
            'dos': ['T1498', 'T1498.001'],
            'ddos': ['T1498', 'T1498.002'],
            'hulk': ['T1498.001'],
            'goldeneye': ['T1498.001'],
            'slowloris': ['T1498.001'],
            'slowhttptest': ['T1498.001'],
            'portscan': ['T1046'],
            'port scan': ['T1046'],
            'scan': ['T1046'],
            'bot': ['T1071', 'T1573'],
            'botnet': ['T1071', 'T1573'],
            'c2': ['T1071', 'T1095'],
            'beacon': ['T1071'],
            'brute': ['T1110.001', 'T1078'],
            'brute force': ['T1110.001', 'T1078'],
            'bruteforce': ['T1110.001', 'T1078'],
            'ftp-patator': ['T1110.001', 'T1071.002'],
            'ssh-patator': ['T1110.001', 'T1021.004'],
            'password': ['T1110.001'],
            'credential': ['T1003', 'T1555'],
            'web attack': ['T1190', 'T1059'],
            'sql': ['T1190'],
            'sqli': ['T1190'],
            'injection': ['T1190'],
            'xss': ['T1190'],
            'phishing': ['T1566'],
            'phish': ['T1566'],
            'infiltration': ['T1041', 'T1071'],
            'exfiltration': ['T1041', 'T1048', 'T1567'],
            'exfil': ['T1041', 'T1048'],
            'data theft': ['T1041', 'T1567'],
            'ransomware': ['T1486'],
            'ransom': ['T1486'],
            'wiper': ['T1561'],
            'defacement': ['T1491'],
            'heartbleed': ['T1212', 'T1190'],
            'exploit': ['T1190', 'T1068'],
            'rdp': ['T1021.001'],
            'smb': ['T1021.002'],
            'psexec': ['T1021.002'],
            'backdoor': ['T1543.003', 'T1136'],
            'persistence': ['T1543.003', 'T1053'],
            'obfuscation': ['T1027'],
            'obfuscated': ['T1027'],
            'encoded': ['T1027'],
            'log clearing': ['T1070'],
            'injection': ['T1055'],
            'archive': ['T1560'],
            'compression': ['T1560'],
            'staging': ['T1005'],
            'powershell': ['T1059'],
            'script': ['T1059'],
            'command': ['T1059'],
            'scheduled task': ['T1053'],
            'privilege escalation': ['T1068', 'T1134'],
            'token': ['T1134'],
            'reconnaissance': ['T1595', 'T1590'],
            'recon': ['T1595'],
            'scanning': ['T1595']
        }

        print(f"✓ Production RAG initialized")
        print(f"  MITRE techniques in KB: {len(self.mitre_kb)}")
        print(f"  Attack keyword mappings: {len(self.technique_mapping)}")

        try:
            import ollama
            response = ollama.list()
            llm_available = any('llama3.1:8b' in m['name'].lower() for m in response.get('models', []))
            if llm_available:
                print(f"  Llama3.1:8b LLM: ✓ Enabled and available")
            else:
                print(f"  Llama3.1:8b LLM: ⚠️  Not found (using template fallback)")
        except:
            print(f"  Llama3.1:8b LLM: ⚠️  Service unavailable")

    def semantic_search(self, query, n_results=3):
        """Perform semantic search in MITRE knowledge base"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )

            retrieved_techniques = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for tech_id in results['ids'][0]:
                    if tech_id in self.mitre_kb:
                        retrieved_techniques.append({
                            'id': tech_id,
                            'data': self.mitre_kb[tech_id]
                        })

            return retrieved_techniques
        except Exception as e:
            print(f"⚠️ Semantic search error: {e}")
            return []

    def keyword_search(self, attack_type):
        """Keyword-based retrieval as fallback"""
        attack_lower = attack_type.lower().strip()
        relevant_techniques = []

        if attack_lower in self.technique_mapping:
            for tech_id in self.technique_mapping[attack_lower]:
                if tech_id in self.mitre_kb:
                    relevant_techniques.append({
                        'id': tech_id,
                        'data': self.mitre_kb[tech_id]
                    })
            return relevant_techniques

        for keyword, tech_ids in self.technique_mapping.items():
            if keyword in attack_lower:
                for tech_id in tech_ids:
                    if tech_id in self.mitre_kb:
                        relevant_techniques.append({
                            'id': tech_id,
                            'data': self.mitre_kb[tech_id]
                        })
                break

        return relevant_techniques

    def extract_flow_features(self, features_dict):
        """Extract key flow features for explanation context"""
        pkt_rate = features_dict.get('Flow Packets/s', 0)
        byte_rate = features_dict.get('Flow Bytes/s', 0)
        duration = features_dict.get('Flow Duration', 0)
        syn_flags = features_dict.get('SYN Flag Count', 0)
        rst_flags = features_dict.get('RST Flag Count', 0)

        flow_desc = []

        if pkt_rate > 1000:
            flow_desc.append(f"{pkt_rate:.0f} pkt/s")
        elif pkt_rate > 100:
            flow_desc.append(f"{pkt_rate:.0f} pkt/s")

        if byte_rate > 1000000:
            flow_desc.append(f"{byte_rate/1000000:.1f} MB/s")
        elif byte_rate > 10000:
            flow_desc.append(f"{byte_rate/1000:.1f} KB/s")

        if syn_flags > 10:
            flow_desc.append(f"{syn_flags} SYN flags")

        if rst_flags > 5:
            flow_desc.append(f"{rst_flags} RST flags")

        if duration > 0 and duration < 100:
            flow_desc.append(f"short duration ({duration:.0f}ms)")

        return ", ".join(flow_desc) if flow_desc else None

    def build_explanation_with_llm(self, attack_type, confidence, mitre_context, features_dict):
        """Build explanation using Llama3.1:8b LLM with MITRE context"""

        if not mitre_context:
            return {
                'explanation': f"{attack_type} attack detected with {confidence:.0%} confidence. Network traffic patterns indicate malicious activity requiring investigation.",
                'mitre_techniques': [],
                'recommended_action': 'Investigate traffic patterns and correlate with threat intelligence',
                'source': 'template_fallback',
                'context_used': 0
            }

        primary = mitre_context[0]
        secondary = mitre_context[1] if len(mitre_context) > 1 else None

        flow_context = self.extract_flow_features(features_dict)
        flow_text = f" ({flow_context})" if flow_context else ""

        mitre_context_str = f"""
PRIMARY TECHNIQUE:
MITRE {primary['id']} - {primary['data']['name']}
Description: {primary['data']['description'][:500]}
Detection: {primary['data']['detection'][0][:200]}
Key Indicators: {', '.join(primary['data']['indicators'][:3])}
"""

        if secondary:
            mitre_context_str += f"""
SECONDARY TECHNIQUE:
MITRE {secondary['id']} - {secondary['data']['name']}
Description: {secondary['data']['description'][:300]}
"""

        prompt = f"""You are a cybersecurity analyst. Analyze this network security detection.

DETECTION SUMMARY:
- Attack Type: {attack_type}
- Confidence: {confidence:.0%}
- Network Behavior: {flow_text.strip('() ')}

RELEVANT MITRE ATT&CK CONTEXT:
{mitre_context_str}

TASK: Provide a brief technical analysis (2-3 sentences) that:
1. Explains what this attack is doing based on the MITRE context
2. States why it was detected (reference specific indicators from MITRE)
3. Suggests one immediate action

CRITICAL REQUIREMENTS:
- Be concise and technical (2-3 sentences maximum)
- Reference MITRE techniques by their IDs (e.g., {primary['id']})
- Focus on actionable security insights
- Do NOT include thinking process or reasoning steps
- Do NOT use conversational language like "Okay", "So", "Let me"
- Provide direct, professional analysis only
- Start with the attack description immediately"""

        try:
            import ollama

            response = ollama.generate(
                model='llama3.1:8b',
                prompt=prompt,
                options={
                    'temperature': 0.2,
                    'num_predict': 1000,
                    'top_p': 0.9,
                    'stop': ['\n\n\n', '<think>', '</think>', 'Note:', 'However,', 'Okay,', 'So,']
                }
            )

            llm_text = response['response'].strip()

            if '<think>' in llm_text:
                llm_text = llm_text.split('<think>')[0].strip()
            if '</think>' in llm_text:
                llm_text = llm_text.split('</think>')[-1].strip()

            conversational_starts = ['Okay,', 'So,', 'Well,', 'Let me', 'I think', 'Hmm,']
            for start in conversational_starts:
                if llm_text.startswith(start):
                    sentences = llm_text.split('. ')
                    if len(sentences) > 2:
                        llm_text = '. '.join(sentences[2:]).strip()
                    break

            if not llm_text or len(llm_text) < 50:
                raise Exception("LLM generated insufficient text")

            mitre_ids = [primary['id']]
            if secondary:
                mitre_ids.append(secondary['id'])

            recommended_action = primary['data']['mitigation'][0]

            return {
                'explanation': llm_text,  # NO TRUNCATION
                'mitre_techniques': mitre_ids,
                'recommended_action': recommended_action,
                'confidence': f'{confidence:.0%}',
                'source': 'rag_llm',
                'context_used': len(mitre_context),
                'llm_used': True
            }

        except Exception as e:
            print(f"⚠️ LLM generation failed: {e}")
            print("   Falling back to template-based explanation...")

            parts = []
            parts.append(f"{attack_type} attack detected with {confidence:.0%} confidence{flow_text}.")

            if secondary:
                parts.append(
                    f"Maps to MITRE {primary['id']} ({primary['data']['name']}) "
                    f"and {secondary['id']} ({secondary['data']['name']})."
                )
            else:
                parts.append(
                    f"Maps to MITRE {primary['id']} ({primary['data']['name']})."
                )

            description = primary['data']['description']
            if '.' in description[:300]:
                first_sentence = description[:description.find('.', 50)+1]
            else:
                first_sentence = description[:250] + "..."
            parts.append(first_sentence)

            indicators = primary['data']['indicators'][:2]
            if indicators:
                parts.append(f"Observed indicators: {' and '.join(indicators)}.")

            explanation = " ".join(parts)

            mitre_ids = [primary['id']]
            if secondary:
                mitre_ids.append(secondary['id'])

            return {
                'explanation': explanation,
                'mitre_techniques': mitre_ids,
                'recommended_action': primary['data']['mitigation'][0],
                'confidence': f'{confidence:.0%}',
                'source': 'template_fallback',
                'context_used': len(mitre_context),
                'llm_used': False
            }

    def explain_with_rag(self, features_dict, attack_type, prediction, confidence):
        """Production RAG explanation with hybrid retrieval"""

        search_query = f"{attack_type} network attack cybersecurity MITRE ATT&CK"
        semantic_results = self.semantic_search(search_query, n_results=3)

        keyword_results = self.keyword_search(attack_type)

        all_results = keyword_results + semantic_results

        seen_ids = set()
        mitre_context = []

        for result in all_results:
            if result['id'] not in seen_ids:
                mitre_context.append(result)
                seen_ids.add(result['id'])
                if len(mitre_context) >= 2:
                    break

        return self.build_explanation_with_llm(
            attack_type, 
            confidence, 
            mitre_context, 
            features_dict
        )

    def batch_explain(self, samples, progress_callback=None):
        """Explain multiple samples efficiently"""
        results = []

        for i, sample in enumerate(samples):
            result = self.explain_with_rag(
                sample['features'],
                sample['attack_type'],
                sample['prediction'],
                sample['confidence']
            )
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, len(samples))

        return results
