# =============================================================================
# MODULE 4 : SIMULATEUR TEMPS RÉEL — DIGITAL TWIN PASPANGA
# =============================================================================
# Simule l'état physique de la station comme si des capteurs renvoyaient
# des mesures toutes les 15 minutes.
#
# Source     : pump_schedule.json (Module 2) — valeurs théoriques
# Sortie     : data/realtime_state.json      — état courant simulé
#              data/realtime_history.json    — historique glissant 24h
#
# Principe :
#   Valeur simulée = valeur théorique + bruit capteur + dérives physiques
#   Les dérives (fuite, panne pompe, encrassement) sont injectables
#   manuellement via data/simulator_events.json
# =============================================================================

import json
import numpy as np
from datetime import datetime, timedelta
import os

# =============================================================================
# PATHS
# =============================================================================

PATH_SCHEDULE  = 'data/pump_schedule.json'
PATH_IFACE     = 'data/module2_interface.json'
PATH_STATE     = 'data/realtime_state.json'
PATH_HISTORY   = 'data/realtime_history.json'
PATH_EVENTS    = 'data/simulator_events.json'   # événements injectables

# =============================================================================
# PARAMÈTRES PHYSIQUES
# =============================================================================

# Volumes réels station Paspanga
V_BACHE_TOTAL  = 6000    # m³  (total bâches source)
V_CHATEAU_UNIT = 2000    # m³  (par château d'eau)
N_CHATEAUX     = 4

DT_HOURS       = 0.25    # 15 min

# Bruit capteur (écart-type, en % ou m³/h)
NOISE_LEVEL_PCT   = 0.8   # ±0.8% sur les niveaux
NOISE_FLOW_M3H    = 3.0   # ±3 m³/h sur les débits
NOISE_POWER_KW    = 2.5   # ±2.5 kW sur la puissance
NOISE_PRESSURE    = 0.05  # ±0.05 bar

# Pression de référence (château plein à 100% → 2.5 bar)
PRESSURE_REF_BAR  = 2.5

# Historique glissant
HISTORY_MAX_SLOTS = 96   # 24h × 4 créneaux/h

# =============================================================================
# UTILITAIRES
# =============================================================================

def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, 'r') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def noise(std):
    """Bruit gaussien centré"""
    return float(np.random.normal(0, std))

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

# =============================================================================
# ÉVÉNEMENTS INJECTABLES
# =============================================================================
# Format simulator_events.json :
# {
#   "fuite_m3h": 15,          # fuite active (0 = aucune)
#   "pompe_en_panne": true,   # force n_pumps=0 quelle que soit la consigne
#   "capteur_bache_ko": false # bloque la valeur bâche (valeur gelée)
# }

DEFAULT_EVENTS = {
    "fuite_m3h":       0,
    "pompe_en_panne":  False,
    "capteur_bache_ko": False,
}

def load_events():
    events = load_json(PATH_EVENTS, DEFAULT_EVENTS)
    # Initialise le fichier si absent
    if not os.path.exists(PATH_EVENTS):
        save_json(PATH_EVENTS, DEFAULT_EVENTS)
    return events

# =============================================================================
# SLOT COURANT
# =============================================================================

def current_slot(schedule):
    """
    Trouve le slot du schedule qui correspond à l'heure courante.
    Supporte les formats "HH:MM" et les timestamps complets.
    """
    now = datetime.now()
    now_str = now.strftime('%H:%M')

    # 1. Tentative de correspondance exacte sur les minutes du fichier
    for s in schedule:
        dt = s.get('datetime', '')
        if dt == now_str or (len(dt) >= 16 and dt[11:16] == now_str):
            return s

    # 2. Fallback robuste calculé au quart d'heure près
    now_minutes = now.hour * 60 + now.minute
    
    def get_slot_total_minutes(slot_item):
        dt = slot_item.get('datetime', '')
        if len(dt) == 5 and ':' in dt:  # Format "HH:MM"
            try:
                h, m = map(int, dt.split(':'))
                return h * 60 + m
            except:
                pass
        elif len(dt) >= 16 and ':' in dt:  # Format "YYYY-MM-DD HH:MM..."
            try:
                h, m = map(int, dt[11:16].split(':'))
                return h * 60 + m
            except:
                pass
        # Fallback ultime si la chaîne est corrompue
        return int(slot_item.get('hour', 0)) * 60

    best = min(schedule, key=lambda s: abs(get_slot_total_minutes(s) - now_minutes))
    return best

# =============================================================================
# SIMULATION D'UN ÉTAT
# =============================================================================

def simulate_state(slot, prev_state, events, iface):
    """
    Calcule l'état simulé courant à partir du slot théorique M2,
    de l'état précédent et des événements actifs.

    Retourne un dict 'realtime_state'.
    """

    rng = np.random.default_rng()   # générateur reproductible

    # ── Récupération valeurs théoriques ──────────────────────────
    n_pumps_th  = slot['n_pumps']
    power_th    = slot['power_kw']
    flow_th     = slot['flow_m3h']
    bache_th    = slot['bache_level_pct']
    chx_th      = slot['chateau_levels_pct']
    chx_mean_th = slot['chateau_mean_pct']
    source      = slot['source']
    demand_th   = slot['flow_conso_m3h']

    # ── Événements ───────────────────────────────────────────────
    fuite       = float(events.get('fuite_m3h', 0))
    panne_pompe = bool(events.get('pompe_en_panne', False))
    capteur_ko  = bool(events.get('capteur_bache_ko', False))

    # ── Pompe en panne → débit nul ───────────────────────────────
    if panne_pompe:
        n_pumps_sim = 0
        flow_sim    = 0.0
        power_sim   = 0.0
    else:
        n_pumps_sim = n_pumps_th
        flow_sim    = max(0, flow_th + noise(NOISE_FLOW_M3H))
        power_sim   = max(0, power_th + noise(NOISE_POWER_KW))

    # ── Bilan hydraulique bâche ──────────────────────────────────
    # Théorie : bache_th vient du module 2 (simulé parfaitement)
    # Réel    : on ajoute bruit + fuite + dérive éventuelle
    if prev_state and not capteur_ko:
        prev_bache = prev_state['bache_level_pct_sim']

        # Apport réseau fixe (50 m³/h)
        Qin       = demand_th + 100
        # Pertes réelles = pompage simulé + fuite
        Qout      = flow_sim + fuite
        delta_pct = (Qin - Qout) * DT_HOURS / V_BACHE_TOTAL * 100
        bache_sim = clamp(prev_bache + delta_pct + noise(NOISE_LEVEL_PCT), 0, 100)

    elif capteur_ko and prev_state:
        # Capteur bloqué → valeur gelée
        bache_sim = prev_state['bache_level_pct_sim']
    else:
        # Premier appel → valeur théorique + bruit
        bache_sim = clamp(bache_th + noise(NOISE_LEVEL_PCT), 0, 100)

    # ── Niveaux châteaux simulés ─────────────────────────────────
    if prev_state:
        prev_chx = prev_state['chateau_levels_pct_sim']
        chx_sim  = []
        for i, c_prev in enumerate(prev_chx):
            # Apport du pompage réparti uniformément
            apport   = flow_sim * DT_HOURS / (V_CHATEAU_UNIT * N_CHATEAUX) * 100
            # Consommation
            conso    = demand_th * DT_HOURS / (V_CHATEAU_UNIT * N_CHATEAUX) * 100
            # Fuite répartie sur tous les châteaux
            perte    = fuite * DT_HOURS / (V_CHATEAU_UNIT * N_CHATEAUX) * 100
            c_new    = clamp(c_prev + apport - conso - perte + noise(NOISE_LEVEL_PCT), 0, 100)
            chx_sim.append(round(c_new, 1))
    else:
        chx_sim = [clamp(c + noise(NOISE_LEVEL_PCT), 0, 100) for c in chx_th]
        chx_sim = [round(c, 1) for c in chx_sim]

    chx_mean_sim = round(float(np.mean(chx_sim)), 1)

    # ── Pression (loi hydrostatique simple) ──────────────────────
    # P = niveau_moyen_château / 100 * P_ref + bruit
    pression_sim = round(
        clamp(chx_mean_sim / 100 * PRESSURE_REF_BAR + noise(NOISE_PRESSURE), 0, 3.5),
        2
    )

    # ── Écarts théorie / réel (pour détection fuite) ─────────────
    delta_bache = round(bache_sim - bache_th, 2)
    delta_flow  = round(flow_sim  - flow_th,  1)
    delta_chx   = round(chx_mean_sim - chx_mean_th, 2)

    # ── Indicateur fuite ─────────────────────────────────────────
    # Si le bilan hydraulique montre une perte anormale
    # (niveau réel < théorique de plus de 3%)
    fuite_suspectee = (
        (fuite > 0 and (delta_bache < -0.05 or delta_chx < -0.05))
        or delta_bache < -3.0
        or delta_chx < -3.0
    )

    # ── Assemblage état ──────────────────────────────────────────
    state = {
        "generated_at":           datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "slot":                   slot['slot'],
        "datetime":               slot['datetime'],
        "hour":                   slot['hour'],

        # Valeurs théoriques (module 2)
        "n_pumps_th":             n_pumps_th,
        "flow_m3h_th":            flow_th,
        "power_kw_th":            power_th,
        "bache_level_pct_th":     bache_th,
        "chateau_mean_pct_th":    chx_mean_th,
        "source":                 source,

        # Valeurs simulées (capteurs)
        "n_pumps_sim":            n_pumps_sim,
        "flow_m3h_sim":           round(flow_sim, 1),
        "power_kw_sim":           round(power_sim, 1),
        "bache_level_pct_sim":    round(bache_sim, 1),
        "chateau_levels_pct_sim": chx_sim,
        "chateau_mean_pct_sim":   chx_mean_sim,
        "pression_bar_sim":       pression_sim,

        # Écarts
        "delta_bache_pct":        delta_bache,
        "delta_flow_m3h":         delta_flow,
        "delta_chateau_pct":      delta_chx,

        # Diagnostics
        "fuite_suspectee":        fuite_suspectee,
        "fuite_active_m3h":       fuite,
        "pompe_en_panne":         panne_pompe,
        "capteur_bache_ko":       capteur_ko,
    }

    return state

# =============================================================================
# MISE À JOUR HISTORIQUE
# =============================================================================

def update_history(state):
    history = load_json(PATH_HISTORY, [])
    history.append(state)
    # Garde seulement les HISTORY_MAX_SLOTS derniers
    if len(history) > HISTORY_MAX_SLOTS:
        history = history[-HISTORY_MAX_SLOTS:]
    save_json(PATH_HISTORY, history)

# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def run_simulator():
    print("\n[MODULE 4] Simulation état temps réel...")

    schedule = load_json(PATH_SCHEDULE)
    iface    = load_json(PATH_IFACE, {})
    events   = load_events()

    if not schedule:
        print("  ✗ pump_schedule.json absent — lancer M2 d'abord")
        return None

    # Override volumes depuis iface si disponibles
    global V_BACHE_TOTAL, V_CHATEAU_UNIT, N_CHATEAUX
    if iface:
        V_BACHE_TOTAL  = iface.get('v_baches_total_m3',  V_BACHE_TOTAL)
        V_CHATEAU_UNIT = iface.get('v_chateau_unit_m3',  V_CHATEAU_UNIT)
        N_CHATEAUX     = iface.get('n_chateaux',         N_CHATEAUX)

    slot      = current_slot(schedule)
    prev_state = load_json(PATH_STATE)

    state = simulate_state(slot, prev_state, events, iface)

    save_json(PATH_STATE, state)
    update_history(state)

    status = []
    if state['fuite_suspectee']:   status.append("⚠ FUITE SUSPECTÉE")
    if state['pompe_en_panne']:    status.append("⚠ PANNE POMPE")
    if state['capteur_bache_ko']:  status.append("⚠ CAPTEUR KO")
    if not status:                 status.append("✓ Normal")

    print(f"  Slot {slot['slot']} | Bâche {state['bache_level_pct_sim']}% "
          f"| Château {state['chateau_mean_pct_sim']}% "
          f"| Δbâche {state['delta_bache_pct']:+.1f}% "
          f"| {' '.join(status)}")

    return state


if __name__ == '__main__':
    run_simulator()
