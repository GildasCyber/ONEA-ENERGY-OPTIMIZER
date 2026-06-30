# =============================================================================
# MODULE 3 : DÉTECTION D'ANOMALIES — PASPANGA
# =============================================================================
# Sources de données : predictions.json + pump_schedule.json (modules 1 & 2)
# Deux couches : règles expertes métier + Isolation Forest ML
# Sortie        : data/anomalies.json
# =============================================================================

import json
import numpy as np
import os
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================

PATH_PREDICTIONS = 'data/predictions.json'
PATH_SCHEDULE    = 'data/pump_schedule.json'
PATH_ANOMALIES   = 'data/anomalies.json'
PATH_REALTIME    = 'data/realtime_state.json'     # module 4

# =============================================================================
# CONSTRUCTION DES DONNÉES DE TRAVAIL
# =============================================================================

def build_records():
    """
    Fusionne predictions.json + pump_schedule.json en une liste de records
    homogènes, utilisables par les deux couches de détection.

    Champs garantis dans chaque record :
        datetime, hour, slot,
        flow_estimated, energy_predicted, solar_capacity_predicted,
        grid_status_predicted, energy_price_kwh,
        n_pumps, power_kw, flow_m3h, source,
        bache_level_pct, chateau_mean_pct, cost_fcfa,
        is_peak_tariff
    """

    with open(PATH_PREDICTIONS, 'r') as f:
        preds = json.load(f)

    with open(PATH_SCHEDULE, 'r') as f:
        sched = json.load(f)

    # index schedule par slot
    sched_by_slot = {s['slot']: s for s in sched}

    records = []

    for p in preds:
        slot = p.get('slot', preds.index(p))
        s    = sched_by_slot.get(slot, {})

        records.append({
            'datetime':                  p.get('datetime', ''),
            'hour':                      p.get('hour', 0),
            'slot':                      slot,

            # Module 1
            'flow_estimated':            p.get('flow_estimated', 0),
            'energy_predicted':          p.get('energy_predicted', 0),
            'solar_capacity_predicted':  p.get('solar_capacity_predicted', 0),
            'grid_status_predicted':     p.get('grid_status_predicted', 1),
            'energy_price_kwh':          p.get('energy_price_kwh', 54),

            # Module 2
            'n_pumps':                   s.get('n_pumps', 0),
            'power_kw':                  s.get('power_kw', 0),
            'flow_m3h':                  s.get('flow_m3h', 0),
            'source':                    s.get('source', 'SONABEL'),
            'bache_level_pct':           s.get('bache_level_pct', 50),
            'chateau_mean_pct':          s.get('chateau_mean_pct', 50),
            'cost_fcfa':                 s.get('cost_fcfa', 0),
            'is_peak_tariff':            s.get('is_peak_tariff', False),
            'generator_kw':              s.get('generator_kw', 0),
            'solar_kw':                  s.get('solar_kw', 0),
        })

    return records


# =============================================================================
# FEATURES POUR ISOLATION FOREST
# =============================================================================

def build_features(records):
    """
    6 features normalisées pour l'Isolation Forest.
    Toutes les valeurs sont numériques et sur des échelles comparables
    après StandardScaler.
    """
    X = []
    for r in records:
        X.append([
            r['flow_estimated'],
            r['energy_predicted'],
            r['bache_level_pct'],
            r['chateau_mean_pct'],
            r['solar_capacity_predicted'],
            r['grid_status_predicted'],
        ])
    return np.array(X, dtype=float)


# =============================================================================
# MODÈLE ML
# =============================================================================

def train_model(records):
    """
    Entraîne sur les records « normaux » uniquement.
    Critères de normalité : réseau dispo, niveaux corrects, débit cohérent.
    Applique StandardScaler pour homogénéiser les échelles.
    """
    normal = [
        r for r in records
        if (
            60  <= r['flow_estimated']           <= 1500  and
            30  <= r['bache_level_pct']           <= 95   and
            30  <= r['chateau_mean_pct']          <= 95   and
            r['grid_status_predicted'] == 1
        )
    ]

    if len(normal) < 5:
        normal = records          # fallback si trop peu de données

    X_train = build_features(normal)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = IsolationForest(
        contamination=0.08,
        random_state=42,
        n_estimators=150
    )
    model.fit(X_scaled)

    print(f"  → Modèle entraîné sur {len(normal)}/{len(records)} observations normales")
    return model, scaler


def detect_ml_anomalies(records, model, scaler):
    """
    Retourne un dict  {slot: {'severity': str, 'score_ml': float}}
    score_ml : valeur brute Isolation Forest (négatif = plus anormal)
    """
    X       = build_features(records)
    X_sc    = scaler.transform(X)
    preds   = model.predict(X_sc)
    scores  = model.score_samples(X_sc)

    result = {}
    for i, (pred, score) in enumerate(zip(preds, scores)):
        if pred == -1:
            result[records[i]['slot']] = {
                'severity': 'CRITIQUE' if score < -0.55 else 'MOYENNE',
                'score_ml': round(float(score), 4),
            }

    print(f"  → ML : {len(result)} anomalies contextuelles détectées")
    return result


# =============================================================================
# RÈGLES EXPERTES
# =============================================================================

def apply_expert_rules(records):
    """
    Retourne un dict  {slot: {'alerts': [...], 'severity_score': int}}
    Chaque alerte incrémente severity_score selon sa gravité.
    Optimisé pour les caractéristiques industrielles de Paspanga (1 pompe = ~400 m3/h).
    """
    result = {}

    for r in records:
        alerts = []
        score  = 0
        slot   = r['slot']

        solar  = r['solar_capacity_predicted']
        grid   = r['grid_status_predicted']
        flow_e = r['flow_estimated']
        flow_p = r['flow_m3h']
        energy = r['energy_predicted']
        hour   = r['hour']
        bache  = r['bache_level_pct']
        chx    = r['chateau_mean_pct']
        ge_kw  = r['generator_kw']
        source = r['source']
        n_p    = r['n_pumps']

        # ── 1. PROTECTION DES ÉQUIPEMENTS (CRITIQUE) ────────────────
        # Risque de marche à sec (Pompage avec bâche vide) : Destructeur pour la pompe
        if n_p > 0 and bache < 15:
            alerts.append("RISQUE_MARCHE_A_SEC_POMPE")
            score += 6

        # Risque de Débordement / Trop-plein des Châteaux d'eau
        if n_p > 0 and chx > 98:
            alerts.append("RISQUE_DEBORDEMENT_CHATEAU")
            score += 3

        # Incohérence Réseau : Planifier du pompage SONABEL alors que le réseau est coupé
        if source == 'SONABEL' and grid == 0 and n_p > 0:
            alerts.append("INCOHERENCE_SOURCE_RESEAU_COUPE")
            score += 4

        # ── 2. OPTIMISATION ÉNERGÉTIQUE ET FINANCIÈRE ───────────────
        # Gaspillage de gasoil : Le groupe tourne alors que le réseau SONABEL est disponible !
        if ge_kw > 0 and grid == 1:
            alerts.append("GASPILLAGE_GASOIL_RESEAU_DISPO")
            score += 5

        # Mauvais rendement du Groupe : Le groupe consomme mais le débit extrait est faible
        if ge_kw > 0 and n_p > 0 and flow_p < (300 * n_p):
            alerts.append("RENDEMENT_GROUPE_ELECTROGENE_FAIBLE")
            score += 3

        # Solaire disponible mais pas exploité (Châteaux non pleins)
        if solar > 60 and n_p == 0 and chx < 85:
            alerts.append("SOLAIRE_NON_EXPLOITE")
            score += 2

        # Pompage lourd en heure de pointe SONABEL (Tarif maximal)
        if r['is_peak_tariff'] and n_p >= 2 and grid == 1:
            alerts.append("POMPAGE_HEURE_POINTE_SONABEL")
            score += 3

        # ── 3. ANOMALIES DE RENDEMENT HYDRAULIQUE ───────────────────
        # Débit anormalement faible pour des pompes actives (1 pompe nominale = ~400 m3/h)
        # Permet de détecter une usure, une vanne à moitié fermée ou un problème technique
        if n_p > 0 and flow_p < (300 * n_p) and ge_kw == 0:
            # Si on est sur le solaire, le débit peut baisser légitimement, on valide avec solar
            if source == 'SOLAR' and solar > 60:
                alerts.append("RENDEMENT_SOLAIRE_ANORMAL")
                score += 2
            elif source == 'SONABEL':
                alerts.append("DEBIT_ANORMAL_POMPE_ACTIVE")
                score += 4

        # Surconsommation énergétique globale (Forte puissance électrique absorbée, faible débit)
        if energy > 200 and flow_p < (250 * n_p) and n_p > 0:
            alerts.append("SURCONSOMMATION_ENERGETIQUE")
            score += 3

        if score > 0:
            result[slot] = {
                'alerts': alerts,
                'severity_score': score,
            }

    print(f"  → Règles expertes optimisées : {len(result)} anomalies détectées")
    return result


# =============================================================================
# FUSION
# =============================================================================

def _severity_label(score):
    if score >= 6:
        return 'CRITIQUE'
    if score >= 3:
        return 'MOYENNE'
    return 'FAIBLE'


def merge_anomalies(records, expert, ml):
    """
    Fusionne les deux sources par slot.
    - Si un slot est détecté par les deux → on enrichit l'entrée règles
      avec le score ML et on majore la sévérité si nécessaire.
    - Si slot ML seul → entrée dédiée COMPORTEMENT_INHABITUEL_ML.
    Score ML converti sur la même échelle que les règles (0–8) :
        score_ml ∈ [-1, 0]  →  severity_score_ml = round((−score_ml) * 8)
    """
    anomalies = []
    slots_done = set()

    for r in records:
        slot = r['slot']
        has_expert = slot in expert
        has_ml     = slot in ml

        if not has_expert and not has_ml:
            continue

        base = {
            'datetime':         r['datetime'],
            'hour':             r['hour'],
            'slot':             slot,
            'flow_estimated':   round(r['flow_estimated'], 1),
            'flow_m3h':         round(r['flow_m3h'], 1),
            'energy_predicted': round(r['energy_predicted'], 1),
            'bache_level_pct':  round(r['bache_level_pct'], 1),
            'chateau_mean_pct': round(r['chateau_mean_pct'], 1),
            'source':           r['source'],
            'n_pumps':          r['n_pumps'],
        }

        if has_expert:
            e = expert[slot]
            score = e['severity_score']
            alerts = list(e['alerts'])
            methods = ['RULE_BASED']

            if has_ml:
                ml_score_raw = ml[slot]['score_ml']
                ml_score_conv = round((-ml_score_raw) * 8)
                score = max(score, ml_score_conv)
                alerts.append('COMPORTEMENT_INHABITUEL_ML')
                methods.append('MACHINE_LEARNING')
                base['ml_anomaly_score'] = ml[slot]['score_ml']

            anomalies.append({
                **base,
                'alerts':            alerts,
                'severity_score':    score,
                'severity':          _severity_label(score),
                'detection_methods': methods,
            })

        elif has_ml:
            ml_score_raw  = ml[slot]['score_ml']
            ml_score_conv = round((-ml_score_raw) * 8)
            anomalies.append({
                **base,
                'alerts':            ['COMPORTEMENT_INHABITUEL_ML'],
                'severity_score':    ml_score_conv,
                'severity':          ml[slot]['severity'],
                'detection_methods': ['MACHINE_LEARNING'],
                'ml_anomaly_score':  ml[slot]['score_ml'],
            })

        slots_done.add(slot)

    anomalies.sort(key=lambda x: x['severity_score'], reverse=True)
    return anomalies


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def detect_anomalies():
    print("\n" + "="*55)
    print("MODULE 3 — Détection anomalies Paspanga")
    print("="*55)

    records = build_records()
    print(f"  → {len(records)} créneaux chargés (predictions + schedule)")

    model, scaler = train_model(records)

    expert = apply_expert_rules(records)
    ml     = detect_ml_anomalies(records, model, scaler)

    # ── Détection fuite via module 4 (si disponible) ──────────────
    realtime = None
    if os.path.exists(PATH_REALTIME):
        with open(PATH_REALTIME, 'r') as f:
            realtime = json.load(f)

    if realtime and realtime.get('fuite_suspectee'):
        slot = realtime.get('slot', 0)
        delta_b = realtime.get('delta_bache_pct', 0)
        delta_c = realtime.get('delta_chateau_pct', 0)
        fuite_score = 6 if min(delta_b, delta_c) < -5 else 4

        existing = expert.get(slot, {'alerts': [], 'severity_score': 0})
        if 'FUITE_PROBABLE' not in existing['alerts']:
            existing['alerts'].append('FUITE_PROBABLE')
            existing['severity_score'] = max(existing['severity_score'], fuite_score)
        expert[slot] = existing
        print(f"  → Fuite suspectée slot {slot} : Δbâche={delta_b:+.1f}% Δchâteau={delta_c:+.1f}%")

    anomalies = merge_anomalies(records, expert, ml)

    output = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'station':      'Paspanga',
        'n_slots':      len(records),
        'anomalies':    anomalies,
    }

    with open(PATH_ANOMALIES, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    n_crit = sum(1 for a in anomalies if a['severity'] == 'CRITIQUE')
    n_moy  = sum(1 for a in anomalies if a['severity'] == 'MOYENNE')
    n_faib = sum(1 for a in anomalies if a['severity'] == 'FAIBLE')

    print(f"\n✓ {len(anomalies)} anomalies → CRITIQUE:{n_crit}  MOYENNE:{n_moy}  FAIBLE:{n_faib}")
    print(f"✓ Exporté → {PATH_ANOMALIES}")
    print("="*55)

    return anomalies


if __name__ == '__main__':
    detect_anomalies()