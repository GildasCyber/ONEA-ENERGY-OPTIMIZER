"""
================================================================================
MODULE 1 : PRÉDICTION ÉNERGÉTIQUE — STATION DE PASPANGA v2.0
================================================================================
Station de pompage de Paspanga — ONEA, Burkina Faso

Paramètres physiques réels :
  - 3 bâches d'eau source : 3 x 2000 m³ = 6000 m³ total
  - 4 pompes : 3 x 90 kW + 1 x 132 kW
  - Débit total : 1750 m³/h (toutes pompes actives)
  - 4 châteaux d'eau destination : 4 x 8000 m³ = 32 000 m³ total
  - Remplissage en série : C1 → C2 → C3 → C4

Architecture v2.0 — résolution native 15 min :
  [CAL-1]  Calendrier perpétuel Burkina Faso via `holidays` (pays BF).
  [CAL-2]  Ramadan calculé via `hijridate` (calendrier hégirien).
  [DAT-1]  Historique glissant 3 ans (1095 jours × 96 pas 15 min).
  [ARC-1]  Rolling Horizon : prédictions 24h × 96 créneaux de 15 min.
  [ARC-2]  Chargement intelligent du modèle (train si absent).
  [PMP-1]  Simulation réaliste des 4 pompes Paspanga.
  [BAC-1]  Modélisation des 3 bâches source (contrainte d'arrêt si vide).
  [ORC-1]  Scheduler APScheduler : run_mpc_iteration() toutes les 15 min.

Unités :
  - energy  → kW  (puissance instantanée, NON des kWh)
  - energy_kwh_15min → kWh consommés sur le créneau (= kW × 0.25 h)
  - flow    → m³/h (débit instantané)
  - solar_capacity → kW (puissance crête disponible)
================================================================================
"""

import numpy as np
import pandas as pd
import json
import os
import logging
from datetime import datetime, timedelta, date

import holidays as holidays_lib
from hijridate import Gregorian

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [MODULE1-PASPANGA] %(levelname)s — %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

os.makedirs('data',   exist_ok=True)
os.makedirs('models', exist_ok=True)

# ============================================================================
# PARAMÈTRES PHYSIQUES — STATION PASPANGA
# ============================================================================

# --- Pompes ---
# 3 pompes de 90 kW + 1 pompe de 132 kW
PUMPS_CONFIG = [
    {'id': 1, 'power_kw': 90,  'flow_m3h': 400, 'label': 'P1-90kW'},
    {'id': 2, 'power_kw': 90,  'flow_m3h': 400, 'label': 'P2-90kW'},
    {'id': 3, 'power_kw': 90,  'flow_m3h': 400, 'label': 'P3-90kW'},
    {'id': 4, 'power_kw': 132, 'flow_m3h': 600, 'label': 'P4-132kW'},
]
# Débit total si toutes actives = 1750 m³/h
N_PUMPS          = len(PUMPS_CONFIG)           # 4
P_TOTAL_MAX      = sum(p['power_kw'] for p in PUMPS_CONFIG)  # 402 kW
Q_TOTAL_MAX      = sum(p['flow_m3h'] for p in PUMPS_CONFIG)  # 1750 m³/h

# Combinaisons de pompes ordonnées par priorité économique
# Ordre logique : démarrer la plus puissante en dernier (économies d'énergie)
# Combinaison 0 pompe, 1 pompe (P1), 2 pompes (P1+P2), 3 pompes (P1+P2+P3), 4 pompes (toutes)
PUMP_COMBINATIONS = [
    {'n': 0, 'pumps': [],        'power_kw': 0,   'flow_m3h': 0,   'label': 'ARRÊT'},
    {'n': 1, 'pumps': [1],       'power_kw': 90,  'flow_m3h': 400, 'label': '1x90kW'},
    {'n': 2, 'pumps': [1, 2],    'power_kw': 180, 'flow_m3h': 800, 'label': '2x90kW'},
    {'n': 3, 'pumps': [1, 2, 3], 'power_kw': 270, 'flow_m3h': 1200, 'label': '3x90kW'},
    {'n': 4, 'pumps': [1, 2, 3, 4], 'power_kw': 402, 'flow_m3h': 1750, 'label': '3x90kW+132kW'},
]

# --- Bâches source ---
N_BACHES        = 3
V_BACHE_UNIT    = 2000      # m³ par bâche
V_BACHES_TOTAL  = N_BACHES * V_BACHE_UNIT   # 6000 m³
BACHE_MIN_PCT   = 10.0      # % minimum pour éviter d'aspirer à sec
BACHE_INIT_PCT  = 70.0      # % initial pour la simulation

# --- Châteaux destination ---
N_CHATEAUX      = 4
V_CHATEAU_UNIT  = 8000      # m³ par château (Capacité globale manipulée par le module 2)
V_CHATEAUX_TOTAL = N_CHATEAUX * V_CHATEAU_UNIT  # 32000 m³
CHATEAU_MIN_PCT = 15.0      # % minimum (service garanti)
CHATEAU_MAX_PCT = 95.0      # % maximum (trop-plein)
CHATEAU_INIT_PCT = [60.0, 55.0, 50.0, 45.0]  # % initiaux C1→C4 (ordre de remplissage)

# Population desservie (estimation ONEA Paspanga)
POPULATION      = 80_000

# --- Tarifs SONABEL — grille MT Type E2 ---
TARIF_CREUSE    = 54        # FCFA/kWh — 00h00–17h00
TARIF_PLEINE    = 118       # FCFA/kWh — 17h00–24h00
HEURE_POINTE    = 17

# --- Groupe électrogène ---
COUT_GE_KWHLITER = 600      # FCFA/L gasoil
RENDEMENT_GE     = 0.33     # kWh/L
TARIF_GE         = 475      # 475 FCFA/kWh

# --- Granularité ---
PAS_PAR_HEURE   = 4                    # 4 créneaux de 15 min par heure
PAS_PAR_JOUR    = 24 * PAS_PAR_HEURE   # 96 créneaux 15 min
DT_MIN          = 15                   # durée d'un pas en minutes
DT_H            = DT_MIN / 60.0        # durée d'un pas en heures (0.25 h)

# --- Rétention historique ---
MAX_RETENTION_DAYS = 1095   # 3 ans

# --- Chemins fichiers ---
PATH_HISTORICAL  = 'data/historical_data.json'
PATH_PREDICTIONS = 'data/predictions.json'
PATH_M2_IFACE    = 'data/module2_interface.json'
PATH_MODEL       = 'models/energy_model_paspanga_v2.pkl'
PATH_METADATA    = 'models/model_metadata_paspanga.json'

# Mots-clés → fête majeure (intensité 2)
_MAJOR_KEYWORDS = [
    'new year', 'independence', 'christmas', 'revolution',
    'eid al-fitr', 'eid al-adha', 'labour', 'proclamation',
]


# ============================================================================
# [CAL-1] CALENDRIER BURKINA FASO — PERPÉTUEL
# ============================================================================

def _classify_holiday_intensity(name: str) -> int:
    name_lower = name.lower()
    if any(kw in name_lower for kw in _MAJOR_KEYWORDS):
        return 2
    return 1


def get_burkina_holidays(year: int) -> dict:
    """[CAL-1] Retourne {date_str: intensité} pour l'année donnée."""
    bf = holidays_lib.country_holidays('BF', years=year)
    result = {}
    for d, name in bf.items():
        date_str  = d.strftime('%Y-%m-%d')
        intensity = _classify_holiday_intensity(name)
        result[date_str] = max(result.get(date_str, 0), intensity)
    return result


# ============================================================================
# [CAL-2] RAMADAN — CALCUL HÉGIRIEN PERPÉTUEL
# ============================================================================

def get_ramadan_period(gregorian_year: int):
    """[CAL-2] Calcule le début et la fin du Ramadan via calendrier hégirien."""
    start_scan = date(gregorian_year - 1, 10, 1)
    start_ram  = None
    end_ram    = None
    for offset in range(700):
        d = start_scan + timedelta(days=offset)
        h = Gregorian(d.year, d.month, d.day).to_hijri()
        if h.year > gregorian_year + 1:
            break
        if h.month == 9 and h.day == 1 and d.year >= gregorian_year - 1:
            if start_ram is None:
                start_ram = d
        if start_ram is not None and h.month == 10 and h.day == 1:
            end_ram = d - timedelta(days=1)
            break
    return start_ram, end_ram


_ramadan_cache: dict = {}

def _get_ramadan_cached(year: int):
    if year not in _ramadan_cache:
        _ramadan_cache[year] = get_ramadan_period(year)
    return _ramadan_cache[year]


def _is_ramadan(dt) -> bool:
    ram_s, ram_e = _get_ramadan_cached(dt.year)
    if ram_s and ram_e:
        d = dt.date() if isinstance(dt, datetime) else dt
        return ram_s <= d <= ram_e
    return False


# ============================================================================
# TARIF HORAIRE
# ============================================================================

def _tarif(hour: int) -> int:
    return TARIF_PLEINE if hour >= HEURE_POINTE else TARIF_CREUSE


# ============================================================================
# [PMP-1] LOGIQUE POMPES PASPANGA
# ============================================================================

def get_pump_combination(n_pumps: int) -> dict:
    """
    [PMP-1] Retourne la configuration de la combinaison pour n_pumps actives.
    n_pumps : 0, 1, 2, 3 ou 4
    """
    for combo in PUMP_COMBINATIONS:
        if combo['n'] == n_pumps:
            return combo
    return PUMP_COMBINATIONS[0]  # ARRÊT par défaut


def select_n_pumps_heuristic(q_needed_m3h: float, bache_level_pct: float) -> int:
    """
    [PMP-1] Sélection heuristique du nombre de pompes selon le débit nécessaire.
    Règle simple : choisir le minimum de pompes qui couvre le débit requis.
    Si les bâches sont presque vides (< BACHE_MIN_PCT) → arrêt.
    """
    if bache_level_pct <= BACHE_MIN_PCT:
        return 0
    for combo in PUMP_COMBINATIONS:
        if combo['flow_m3h'] >= q_needed_m3h:
            return combo['n']
    return 4  # maximum si débit requis dépasse toutes les combos


# ============================================================================
# PATTERN CONSOMMATION EAU — PASPANGA
# ============================================================================

def get_water_consumption_pattern(hour, is_ramadan=False, holiday_intensity=0,
                                   temp_ext=35.0, is_weekend=False, minute=0):
    """
    Consommation en m³/h pour un créneau de 15 min à Paspanga (~80 000 pers).
    La valeur retournée est un **débit instantané en m³/h** (pas un volume),
    cohérent avec les débits de pompes (m³/h).
    Moyenne journalière ≈ 100 m³/h → ~2400 m³/jour.
    Interpolation linéaire intra-heure sur les 4 quarts de 15 min.
    """
    base_pattern = {
        0: 0.30, 1: 0.25, 2: 0.22, 3: 0.22, 4: 0.25, 5: 0.40,
        6: 0.85, 7: 1.20, 8: 1.35, 9: 1.15, 10: 0.90, 11: 0.85,
        12: 0.95, 13: 0.88, 14: 0.78, 15: 0.82, 16: 0.95, 17: 1.10,
        18: 1.30, 19: 1.40, 20: 1.35, 21: 1.15, 22: 0.78, 23: 0.50,
    }
    Q_ref = 800.0   # m³/h de référence (débit nominal toutes pompes)

    # Interpolation linéaire entre l'heure courante et la suivante
    alpha  = minute / 60.0
    f_cur  = base_pattern[hour]
    f_next = base_pattern[(hour + 1) % 24]
    factor = f_cur + alpha * (f_next - f_cur)

    # Température : chaleur augmente la consommation
    if temp_ext > 42:
        factor *= 1.25
    elif temp_ext > 38:
        factor *= 1.15

    # Weekend : légère hausse
    if is_weekend:
        factor *= 1.08

    # Jours fériés
    if holiday_intensity == 2:
        factor *= 1.20
    elif holiday_intensity == 1:
        factor *= 1.10

    # Ramadan : Sahur (02h–04h) et Iftar (18h–20h)
    if is_ramadan:
        if hour in [2, 3, 4]:
            factor *= 1.40
        elif hour in [18, 19, 20]:
            factor *= 1.50
        elif 5 <= hour <= 17:
            factor *= 0.75

    return Q_ref * factor   # m³/h (débit instantané)


# ============================================================================
# [DAT-1] GESTION HISTORIQUE GLISSANT 3 ANS
# ============================================================================

def _load_historical() -> list:
    if not os.path.exists(PATH_HISTORICAL):
        return []
    with open(PATH_HISTORICAL, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save_historical(data: list):
    data = _purge_old_entries(data)
    with open(PATH_HISTORICAL, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _purge_old_entries(data: list) -> list:
    if not data:
        return data
    cutoff = (datetime.now() - timedelta(days=MAX_RETENTION_DAYS)).strftime('%Y-%m-%d')
    original_len = len(data)
    data = [row for row in data if row.get('date', '9999') >= cutoff]
    purged = original_len - len(data)
    if purged > 0:
        log.info(f"Purge historique : {purged} enregistrements supprimés (> {MAX_RETENTION_DAYS} jours)")
    return data


def log_real_time_data(entry: dict):
    """
    [DAT-1] Ajoute une mesure réelle à historical_data.json.

    Champs attendus :
      date, hour, minute (nouveau), energy (kW), flow (m³/h),
      bache_level_pct, temp_ext, humidity, solar_capacity (kW),
      is_grid_available, current_source, n_pumps_active, energy_price_kwh.
    """
    data = _load_historical()
    if 'day_of_week' not in entry:
        try:
            dt = datetime.strptime(entry['date'], '%Y-%m-%d')
            entry['day_of_week']       = dt.weekday()
            entry['month']             = dt.month
            entry['is_weekend']        = int(dt.weekday() >= 5)
            entry['is_end_of_month']   = int(dt.day >= 25)
            entry['is_ramadan_period'] = int(_is_ramadan(dt))
            all_h = get_burkina_holidays(dt.year)
            entry['holiday_intensity'] = all_h.get(entry['date'], 0)
            entry['is_solar_window']   = int(7 <= entry.get('hour', 0) <= 17)
            entry.setdefault('minute', 0)
            entry.setdefault('ramadan_hour_shift', 0)
            entry.setdefault('temp_humidity_interaction',
                             entry.get('temp_ext', 30) * (entry.get('humidity', 40) / 100))
            entry.setdefault('energy_price_kwh', 0)
            entry.setdefault('holiday_window', 0)
        except Exception as exc:
            log.warning(f"log_real_time_data : complétion partielle — {exc}")
    data.append(entry)
    _save_historical(data)
    log.info(
        f"Mesure réelle enregistrée : {entry.get('date')} "
        f"{entry.get('hour', '?'):02}h{entry.get('minute', 0):02d} "
        f"| {entry.get('energy', '?')} kW | {entry.get('flow', '?')} m³/h"
    )


# ============================================================================
# [BAC-1] SIMULATION BÂCHES SOURCE
# ============================================================================

def simulate_bache_dynamics(bache_level_pct: float, q_pomped_m3h: float,
                              q_treated_inflow_m3h: float, dt_h: float) -> float:
    """
    [BAC-1] Simule la dynamique d'une bâche source sur un pas de temps.

    Bilan : V_baches x Δlevel = (q_treated_inflow - q_pomped) x dt
    L'eau traitée entre dans les bâches, les pompes en prennent.
    Arrêt forcé si niveau < BACHE_MIN_PCT.

    Retourne le nouveau niveau (%) en [BACHE_MIN_PCT, 100].
    """
    delta_vol = (q_treated_inflow_m3h - q_pomped_m3h) * dt_h
    delta_pct = (delta_vol / V_BACHES_TOTAL) * 100.0
    new_level = float(np.clip(bache_level_pct + delta_pct, BACHE_MIN_PCT, 100.0))
    return new_level


# ============================================================================
# GÉNÉRATION DONNÉES HISTORIQUES INITIALES (365 jours)
# ============================================================================

def generate_yearly_data(num_days=365):
    """
    Génère num_days jours de données simulées en pas de 15 min (96 créneaux/jour).

    UNITÉS :
      - energy  → puissance instantanée en **kW** (ce que consomment les pompes)
      - flow    → débit en **m³/h** (débit instantané)
      - bache_level_pct → niveau % des bâches

    La dynamique bâches utilise DT_H = 0.25 h pour convertir débit → volume.
    """
    log.info(f"Génération de {num_days} jours × 96 pas (15 min) — station Paspanga...")

    data       = []
    start_date = datetime.now() - timedelta(days=num_days)

    bache_level_pct = BACHE_INIT_PCT

    # Pré-chargement fériés
    all_holidays = {}
    for year in range(start_date.year, datetime.now().year + 2):
        all_holidays.update(get_burkina_holidays(year))

    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        date_str     = current_date.strftime('%Y-%m-%d')
        month        = current_date.month
        dow          = current_date.weekday()
        is_weekend   = dow >= 5

        if month in [3, 4, 5]:
            temp_base = np.random.normal(38, 2.5)
        elif month in [6, 7, 8, 9]:
            temp_base = np.random.normal(30, 2.0)
        else:
            temp_base = np.random.normal(33, 2.0)

        is_ram      = _is_ramadan(current_date)
        holiday_int = all_holidays.get(date_str, 0)

        yesterday = (current_date - timedelta(days=1)).strftime('%Y-%m-%d')
        tomorrow  = (current_date + timedelta(days=1)).strftime('%Y-%m-%d')
        if   date_str in all_holidays:  holiday_window = 0
        elif tomorrow in all_holidays:  holiday_window = -1
        elif yesterday in all_holidays: holiday_window = 1
        else:                           holiday_window = 0

        is_end_of_month = current_date.day >= 25

        for slot in range(PAS_PAR_JOUR):                    # 96 créneaux
            hour   = slot // PAS_PAR_HEURE                  # 0–23
            minute = (slot % PAS_PAR_HEURE) * DT_MIN        # 0, 15, 30, 45

            # ── Température horaire (interpolée) ──────────────────────────────
            if   0  <= hour <= 6:  temp_ext = temp_base - np.random.uniform(7, 11)
            elif 7  <= hour <= 12: temp_ext = temp_base - np.random.uniform(1, 4)
            elif 13 <= hour <= 16: temp_ext = temp_base + np.random.uniform(0, 3)
            else:                  temp_ext = temp_base - np.random.uniform(2, 6)
            temp_ext = float(np.clip(temp_ext, 18, 46))

            # ── Humidité ──────────────────────────────────────────────────────
            if   month in [6, 7, 8, 9]: humidity = float(np.random.uniform(58, 88))
            elif month in [3, 4, 5]:    humidity = float(np.random.uniform(12, 35))
            else:                        humidity = float(np.random.uniform(18, 45))
            temp_humidity_interaction = temp_ext * (humidity / 100)

            # ── Solaire (kW crête) ────────────────────────────────────────────
            is_solar_window = 7 <= hour <= 17
            solar_capacity  = 0.0
            if is_solar_window:
                cloud_factor   = max(0.0, 1.0 - humidity / 100 * 0.35)
                base_solar     = 90 if 10 <= hour <= 15 else (50 if hour in [8, 9, 16, 17] else 25)
                solar_capacity = float(max(0.0, base_solar * cloud_factor + np.random.normal(0, 6)))

            # ── Disponibilité réseau SONABEL ──────────────────────────────────
            is_grid_available = 1
            if   temp_ext > 40 and np.random.random() > 0.82:        is_grid_available = 0
            elif 18 <= hour <= 22 and np.random.random() > 0.88:     is_grid_available = 0
            elif np.random.random() > 0.96:                          is_grid_available = 0

            # ── Source et prix ────────────────────────────────────────────────
            if solar_capacity > 20:
                current_source   = 0
                energy_price_kwh = 0
            elif is_grid_available:
                current_source   = 1
                energy_price_kwh = _tarif(hour)
            else:
                current_source   = 2
                energy_price_kwh = TARIF_GE

            # ── Ramadan ───────────────────────────────────────────────────────
            ramadan_hour_shift = 0
            if is_ram:
                if   hour in [2, 3, 4]:  ramadan_hour_shift = 1
                elif hour in [18, 19]:   ramadan_hour_shift = 2

            # ── Consommation eau réseau (m³/h, débit instantané) ─────────────
            Q_conso = float(max(15.0,
                get_water_consumption_pattern(hour, is_ram, holiday_int,
                                               temp_ext, is_weekend, minute)
                + np.random.normal(0, 8)))

            # ── Choix du nombre de pompes ─────────────────────────────────────
            bache_error    = bache_level_pct - 70.0
            q_target       = float(np.clip(Q_conso - bache_error * 2.0, 0, Q_TOTAL_MAX))
            n_pumps_active = select_n_pumps_heuristic(q_target, bache_level_pct)
            combo          = get_pump_combination(n_pumps_active)

            q_pomped  = float(combo['flow_m3h'])             # m³/h
            # energy = puissance instantanée en kW (PAS en kWh)
            energy_kw = float(max(0.0, combo['power_kw'] + np.random.normal(0, 5)))

            # ── Dynamique bâches (DT_H = 0.25 h) ─────────────────────────────
            q_treated_inflow = float(max(0.0, Q_conso * 1.05 + np.random.normal(0, 5)))
            bache_level_pct  = simulate_bache_dynamics(
                bache_level_pct, q_pomped, q_treated_inflow, DT_H)

            data.append({
                'date':                      date_str,
                'hour':                      hour,
                'minute':                    minute,
                'day_of_week':               dow,
                'month':                     month,
                'is_weekend':                int(is_weekend),
                'holiday_intensity':         holiday_int,
                'holiday_window':            holiday_window,
                'is_end_of_month':           int(is_end_of_month),
                'is_ramadan_period':         int(is_ram),
                'ramadan_hour_shift':        ramadan_hour_shift,
                'temp_ext':                  round(temp_ext, 1),
                'humidity':                  round(humidity, 1),
                'temp_humidity_interaction': round(temp_humidity_interaction, 2),
                'current_source':            current_source,
                'is_grid_available':         is_grid_available,
                'energy_price_kwh':          energy_price_kwh,
                'is_solar_window':           int(is_solar_window),
                'solar_capacity':            round(solar_capacity, 2),  # kW
                'flow':                      round(Q_conso, 2),          # m³/h
                'energy':                    round(energy_kw, 2),        # kW (puissance)
                'n_pumps_active':            n_pumps_active,
                'bache_level_pct':           round(bache_level_pct, 2),
            })

    log.info(f"✓ {len(data)} enregistrements générés ({num_days} jours × 96 pas 15 min)")
    return data


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

FEATURE_COLS = [
    # Temporelles
    'hour', 'day_of_week', 'month', 'is_weekend',
    'holiday_intensity', 'holiday_window', 'is_end_of_month',
    'is_ramadan_period', 'ramadan_hour_shift',
    # Météo
    'temp_ext', 'humidity', 'temp_humidity_interaction',
    # Technique
    'current_source', 'is_grid_available', 'energy_price_kwh', 'is_solar_window',
    # Lags
    'energy_lag_1h', 'energy_lag_24h', 'energy_lag_7days',
    'flow_lag_1h', 'solar_lag_1h',
    # Rolling
    'energy_mean_3h', 'energy_mean_24h', 'energy_std_3h',
]


def add_lag_and_rolling_features(df):
    """
    Calcule les features de lag et rolling adaptées aux pas de 15 min.

    Correspondances temporelles (1 pas = 15 min) :
      lag_1step   = 15 min
      lag_4steps  = 1 h
      lag_96steps = 24 h (même créneau hier)
      lag_672steps= 7 jours (même créneau -7j)

    Rolling windows :
      rolling_12  = 3 h  (12 × 15 min)
      rolling_96  = 24 h (96 × 15 min)
    """
    log.info("Calcul features LAG et ROLLING (résolution 15 min)...")
    df = df.sort_values(['date', 'hour', 'minute']).reset_index(drop=True)

    # Lags temporels
    df['energy_lag_1h']    = df['energy'].shift(4)    # 4 pas = 1 h
    df['energy_lag_24h']   = df['energy'].shift(96)   # 96 pas = 24 h
    df['energy_lag_7days'] = df['energy'].shift(672)  # 672 pas = 7 j
    df['flow_lag_1h']      = df['flow'].shift(4)
    df['solar_lag_1h']     = df['solar_capacity'].shift(4)

    # Rolling (3 h = 12 pas, 24 h = 96 pas)
    df['energy_mean_3h']  = df['energy'].rolling(12, min_periods=1).mean()
    df['energy_mean_24h'] = df['energy'].rolling(96, min_periods=1).mean()
    df['energy_std_3h']   = df['energy'].rolling(12, min_periods=1).std().fillna(0)

    for col in ['energy_lag_1h', 'energy_lag_24h', 'energy_lag_7days',
                'flow_lag_1h', 'solar_lag_1h']:
        df[col] = df[col].ffill().bfill()

    log.info("✓ Features LAG/ROLLING 15 min ajoutées")
    return df


def handle_outliers(df, columns):
    log.info("Gestion valeurs aberrantes (3σ)...")
    count = 0
    for col in columns:
        mean, std = df[col].mean(), df[col].std()
        before = ((df[col] < mean - 3 * std) | (df[col] > mean + 3 * std)).sum()
        df[col] = df[col].clip(mean - 3 * std, mean + 3 * std)
        count  += before
    log.info(f"✓ {count} valeurs aberrantes corrigées")
    return df


# ============================================================================
# ENTRAÎNEMENT MODÈLE
# ============================================================================

def train_model(df):
    log.info("=" * 70)
    log.info("ENTRAÎNEMENT MODÈLE ML — STATION PASPANGA v1.0")
    log.info("=" * 70)

    df_clean = df.dropna(subset=FEATURE_COLS + ['energy']).copy()
    X, y     = df_clean[FEATURE_COLS], df_clean['energy']

    log.info(f"Données : {len(X)} samples, {len(FEATURE_COLS)} features")
    log.info(f"Période : {df_clean['date'].iloc[0]} → {df_clean['date'].iloc[-1]}")

    split_idx       = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    log.info(f"Split : {len(X_train)} train / {len(X_test)} test")

    tscv       = TimeSeriesSplit(n_splits=5)
    param_grid = {
        'n_estimators':      [100, 150],
        'max_depth':         [10, 15, None],
        'min_samples_split': [5, 10],
        'min_samples_leaf':  [2, 4],
        'max_features':      ['sqrt'],
    }
    log.info("GridSearchCV (sans data leakage)...")
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid, cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1, verbose=0,
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    log.info(f"✓ Meilleurs params : {best_params}")

    best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)
    mask   = y_test > 1.0
    mape   = float(np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100) if mask.any() else 0.0
    log.info(f"Métriques TEST — MAE:{mae:.2f} kWh | RMSE:{rmse:.2f} | R²:{r2:.4f} | MAPE:{mape:.2f}%")

    # Réentraîner sur 100% pour déploiement
    best_model.fit(X, y)
    log.info("✓ Modèle réentraîné sur 100% pour déploiement")

    fi = pd.DataFrame({
        'feature':    FEATURE_COLS,
        'importance': best_model.feature_importances_,
    }).sort_values('importance', ascending=False)
    log.info("Top 10 features :\n" +
             "\n".join(f"  {r['feature']:35s}: {r['importance']:.4f}"
                       for _, r in fi.head(10).iterrows()))

    joblib.dump(best_model, PATH_MODEL)
    metadata = {
        'version':       'paspanga-1.0',
        'station':       'Paspanga',
        'date_trained':  datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_samples':   len(X),
        'num_features':  len(FEATURE_COLS),
        'feature_cols':  FEATURE_COLS,
        'best_params':   best_params,
        'test_metrics':  {'mae': round(mae, 3), 'rmse': round(rmse, 3),
                          'r2': round(r2, 4),   'mape': round(mape, 2)},
        'tarifs':        {'creuse_fcfa': TARIF_CREUSE, 'pleine_fcfa': TARIF_PLEINE,
                          'ge_fcfa': TARIF_GE,         'heure_pointe': HEURE_POINTE},
        'feature_importance': fi.to_dict('records'),
        'system_params': {
            'pumps_config':     PUMPS_CONFIG,
            'n_pumps':          N_PUMPS,
            'q_total_max_m3h':  Q_TOTAL_MAX,
            'p_total_max_kw':   P_TOTAL_MAX,
            'n_baches':         N_BACHES,
            'v_baches_total_m3': V_BACHES_TOTAL,
            'n_chateaux':       N_CHATEAUX,
            'v_chateaux_total_m3': V_CHATEAUX_TOTAL,
            'population':       POPULATION,
        },
    }
    with open(PATH_METADATA, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    log.info(f"✓ Modèle sauvegardé : {PATH_MODEL}")
    log.info(f"✓ Métadonnées       : {PATH_METADATA}")
    return best_model


# ============================================================================
# [ARC-2] CHARGEMENT INTELLIGENT DU MODÈLE
# ============================================================================

def load_or_train_model():
    """[ARC-2] Charge le modèle si existant, sinon entraîne."""
    if os.path.exists(PATH_MODEL):
        log.info(f"✓ Modèle chargé depuis {PATH_MODEL}")
        return joblib.load(PATH_MODEL)
    log.info("Modèle absent → entraînement complet...")
    data = _load_historical()
    if not data:
        log.info("Historique absent → génération initiale 365 jours")
        data = generate_yearly_data(365)
        _save_historical(data)
        with open(PATH_HISTORICAL, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    df = pd.DataFrame(data)
    df = add_lag_and_rolling_features(df)
    df = handle_outliers(df, ['energy', 'flow', 'temp_ext'])
    return train_model(df)


# ============================================================================
# [ARC-1] EXTRACTION DES LAGS DEPUIS HISTORIQUE
# ============================================================================

def _extract_lags_from_history(historical_df: pd.DataFrame, step: int,
                                predictions_so_far: list) -> dict:
    """
    [ARC-1] Extrait les features de lag en résolution 15 min.

    Règles de lookup (1 pas = 15 min) :
      lag_1h    → 4 pas en arrière dans predictions_so_far ou historique
      lag_24h   → 96 pas en arrière (même créneau hier)
      lag_7days → 672 pas en arrière (même créneau -7j)
      rolling_3h  → moyenne des 12 derniers pas disponibles
      rolling_24h → moyenne des 96 derniers pas disponibles
    """
    n_hist = len(historical_df)

    def _get_past_energy(steps_back: int) -> float:
        """Retourne l'énergie du pas -steps_back (prédit ou historique)."""
        if step > steps_back:
            return float(predictions_so_far[step - steps_back - 1]['energy_predicted'])
        hist_idx = n_hist - (steps_back - step + 1)
        hist_idx = max(0, hist_idx)
        return float(historical_df.iloc[hist_idx]['energy'])

    def _get_past_flow(steps_back: int) -> float:
        if step > steps_back:
            return float(predictions_so_far[step - steps_back - 1]['flow_demand_forecast'])
        hist_idx = max(0, n_hist - (steps_back - step + 1))
        return float(historical_df.iloc[hist_idx]['flow'])

    def _get_past_solar(steps_back: int) -> float:
        if step > steps_back:
            return float(predictions_so_far[step - steps_back - 1]['solar_capacity_predicted'])
        hist_idx = max(0, n_hist - (steps_back - step + 1))
        return float(historical_df.iloc[hist_idx].get('solar_capacity', 0))

    energy_lag_1h    = _get_past_energy(4)    # 1 h = 4 pas
    energy_lag_24h   = _get_past_energy(96)   # 24 h = 96 pas
    energy_lag_7days = _get_past_energy(672)  # 7 j  = 672 pas
    flow_lag_1h      = _get_past_flow(4)
    solar_lag_1h     = _get_past_solar(4)

    # Rolling 3 h (12 pas) — on mixe prédit + historique si nécessaire
    window_3h  = min(12, step + n_hist)
    window_24h = min(96, step + n_hist)

    recent_3h = [
        float(predictions_so_far[step - i - 1]['energy_predicted'])
        if step > i else
        float(historical_df.iloc[max(0, n_hist - (i - step + 1))]['energy'])
        for i in range(window_3h)
    ]
    recent_24h = [
        float(predictions_so_far[step - i - 1]['energy_predicted'])
        if step > i else
        float(historical_df.iloc[max(0, n_hist - (i - step + 1))]['energy'])
        for i in range(window_24h)
    ]

    energy_mean_3h  = float(np.mean(recent_3h))  if recent_3h  else 0.0
    energy_std_3h   = float(np.std(recent_3h))   if len(recent_3h) > 1 else 0.0
    energy_mean_24h = float(np.mean(recent_24h)) if recent_24h else 0.0

    return {
        'energy_lag_1h':    energy_lag_1h,
        'energy_lag_24h':   energy_lag_24h,
        'energy_lag_7days': energy_lag_7days,
        'flow_lag_1h':      flow_lag_1h,
        'solar_lag_1h':     solar_lag_1h,
        'energy_mean_3h':   energy_mean_3h,
        'energy_mean_24h':  energy_mean_24h,
        'energy_std_3h':    energy_std_3h,
    }


# ============================================================================
# CONSTRUCTION INTERFACE MODULE 2
# ============================================================================

def _build_module2_interface(predictions: list, historical_df: pd.DataFrame = None):
    """
    Construit data/module2_interface.json pour le Module 2 d'optimisation.

    Horizon : 96 créneaux de 15 min (24 h glissantes).

    Grandeurs transmises :
      conso_base_kw      : puissance pompage prédite [kW] par créneau
      conso_base_kwh     : énergie par créneau (kW × 0.25 h) [kWh]
      solaire_kw         : puissance solaire disponible [kW]
      flow_demand_m3h    : débit demandé [m³/h]
      bache_levels_pct   : évolution niveau bâches sur l'horizon [%]
      stress_hydraulique : 1 si débit > seuil critique

    Cohérence physique :
      Le Module 2 peut déduire le volume pompé sur un créneau :
        vol_m3 = flow_m3h × DT_H  (avec DT_H = 0.25 h)
    """
    # Niveau bâches initial depuis historique
    if historical_df is not None and len(historical_df) > 0:
        bache_level = float(historical_df.iloc[-1].get('bache_level_pct', BACHE_INIT_PCT))
    else:
        hist = _load_historical()
        bache_level = float(hist[-1].get('bache_level_pct', BACHE_INIT_PCT)) if hist else BACHE_INIT_PCT

    log.info(f"Interface Module 2 — niveau bâches initial : {bache_level:.1f}%")

    labels_15min       = []
    tarifs_15min       = []
    conso_kw           = []    # puissance instantanée [kW]
    conso_kwh          = []    # énergie par pas 15 min [kWh]
    solar_kw           = []    # puissance solaire [kW]
    flow_demand_m3h    = []    # débit demandé [m³/h]
    bache_levels_pct   = []    # évolution niveau bâches [%]
    stress_hydraulique = []
    current_source_list = []

    for pred in predictions:
        hour      = pred['hour']
        minute    = pred['minute']
        dt_str    = pred.get('datetime', f"{pred['date']} {hour:02d}:{minute:02d}")

        labels_15min.append(dt_str[-5:])    # "HH:MM"
        tarifs_15min.append(TARIF_PLEINE if hour >= HEURE_POINTE else TARIF_CREUSE)
        conso_kw.append(round(pred['energy_predicted'], 2))
        conso_kwh.append(round(pred.get('energy_kwh_15min',
                                         pred['energy_predicted'] * DT_H), 4))
        solar_kw.append(round(pred['solar_capacity_predicted'], 2))
        flow_demand_m3h.append(round(pred['flow_demand_forecast'], 2))
        bache_levels_pct.append(round(pred.get('bache_level_pct', bache_level), 2))
        stress_hydraulique.append(1 if pred['flow_demand_forecast'] > 1300.0 else 0)
        current_source_list.append(pred['current_source'])

    # Énergie totale 24 h = somme des kWh par créneau
    energie_totale_24h_kwh = round(sum(conso_kwh), 2)

    interface = {
        'generated_at':            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'station':                 'Paspanga',
        'date':                    predictions[0]['date'],
        'horizon_steps':           PAS_PAR_JOUR,          # 96
        'dt_minutes':              DT_MIN,                 # 15
        'dt_hours':                DT_H,                  # 0.25

        # ── Séries temporelles 96 créneaux ───────────────────────────────────
        'labels':                  labels_15min,           # ["HH:MM", ...]
        'tarifs':                  tarifs_15min,           # FCFA/kWh par créneau
        'conso_base_kw':           conso_kw,               # puissance [kW]
        'conso_base_kwh':          conso_kwh,              # énergie par pas [kWh]
        'solaire_kw':              solar_kw,               # [kW]
        'flow_demand_m3h':         flow_demand_m3h,        # [m³/h]
        'bache_levels_pct':        bache_levels_pct,       # [%] évolution
        'stress_hydraulique':      stress_hydraulique,
        'current_source':          current_source_list,

        # Alias legacy (compatibilité module2_optimisation existant)
        'conso_base':              conso_kw,               # [kW] — NB: était kWh, corrigé
        'solaire':                 solar_kw,
        'flow_demand_forecast':    flow_demand_m3h,

        # ── Résumés ──────────────────────────────────────────────────────────
        'energie_totale_24h_kwh':  energie_totale_24h_kwh,

        # ── Paramètres tarifaires ─────────────────────────────────────────────
        'tarif_creuse':            TARIF_CREUSE,
        'tarif_pleine':            TARIF_PLEINE,
        'tarif_ge':                TARIF_GE,
        'heure_pointe':            HEURE_POINTE,

        # ── Pompes Paspanga ───────────────────────────────────────────────────
        'pump_combinations':       PUMP_COMBINATIONS,
        'p_total_max_kw':          P_TOTAL_MAX,
        'q_total_max_m3h':         Q_TOTAL_MAX,

        # ── Bâches source ─────────────────────────────────────────────────────
        'bache_level_init_pct':    bache_level,
        'v_baches_total_m3':       V_BACHES_TOTAL,
        'n_baches':                N_BACHES,
        'bache_min_pct':           BACHE_MIN_PCT,

        # ── Châteaux destination ──────────────────────────────────────────────
        'chateau_levels_init':     CHATEAU_INIT_PCT,
        'v_chateau_unit_m3':       V_CHATEAU_UNIT,
        'n_chateaux':              N_CHATEAUX,
        'chateau_min_pct':         CHATEAU_MIN_PCT,
        'chateau_max_pct':         CHATEAU_MAX_PCT,
    }

    with open(PATH_M2_IFACE, 'w', encoding='utf-8') as f:
        json.dump(interface, f, indent=2, ensure_ascii=False)
    log.info(
        f"✓ Interface Module 2 : {PATH_M2_IFACE} "
        f"({len(labels_15min)} créneaux × 15 min | "
        f"Σ énergie 24h = {energie_totale_24h_kwh} kWh)"
    )


# ============================================================================
# [ARC-1] PRÉDICTIONS ROLLING HORIZON — 24h x 15 min
# ============================================================================

def make_predictions(model, start_time: datetime = None) -> list:
    """
    [ARC-1] Rolling Horizon MPC — 96 pas de 15 min (= 24 h glissantes).

    Alignement sur la grille 15 min :
      start_time est arrondi au prochain créneau 15 min.

    Chaque pas produit :
      - energy_predicted    : puissance prévue en **kW** (instantanée)
      - energy_kwh_15min    : énergie consommée sur le pas (kW × 0.25 h = kWh)
      - flow_demand_forecast: débit demandé en m³/h
      - bache_level_pct     : niveau bâches simulé dynamiquement (%)

    La dynamique bâches est réintégrée dans la boucle de prédiction pour
    fournir au module 2 un état initial cohérent à chaque pas.
    """
    if start_time is None:
        start_time = datetime.now()

    # Aligner sur la grille 15 min
    remainder = start_time.minute % DT_MIN
    if remainder:
        start_time = start_time + timedelta(minutes=DT_MIN - remainder)
    start_time = start_time.replace(second=0, microsecond=0)

    log.info("=" * 70)
    log.info(f"PRÉDICTIONS ROLLING HORIZON 96 × 15 min — départ : {start_time.strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 70)

    historical_data = _load_historical()
    if not historical_data:
        raise RuntimeError("Historique vide — lancez d'abord generate_yearly_data().")
    historical_df = pd.DataFrame(historical_data).sort_values(
        ['date', 'hour', 'minute'] if 'minute' in historical_data[0] else ['date', 'hour']
    )

    # Niveau initial des bâches depuis le dernier enregistrement réel
    bache_level_pct = float(historical_df.iloc[-1].get('bache_level_pct', BACHE_INIT_PCT))

    predictions: list = []

    for step in range(PAS_PAR_JOUR):                         # 96 pas
        slot_time = start_time + timedelta(minutes=step * DT_MIN)
        hour      = slot_time.hour
        minute    = slot_time.minute
        month     = slot_time.month
        dow       = slot_time.weekday()
        is_weekend      = dow >= 5
        is_end_of_month = slot_time.day >= 25
        slot_date_str   = slot_time.strftime('%Y-%m-%d')

        # ── Calendrier ────────────────────────────────────────────────────────
        all_h             = get_burkina_holidays(slot_time.year)
        holiday_intensity = all_h.get(slot_date_str, 0)
        yesterday_str     = (slot_time - timedelta(days=1)).strftime('%Y-%m-%d')
        tomorrow_str      = (slot_time + timedelta(days=1)).strftime('%Y-%m-%d')
        if   slot_date_str in all_h: holiday_window = 0
        elif tomorrow_str  in all_h: holiday_window = -1
        elif yesterday_str in all_h: holiday_window = 1
        else:                        holiday_window = 0

        is_ram = _is_ramadan(slot_time)

        # ── Météo ─────────────────────────────────────────────────────────────
        if   month in [3, 4, 5]:    temp_base = 37
        elif month in [6, 7, 8, 9]: temp_base = 30
        else:                        temp_base = 33

        if   0  <= hour <= 5:  temp_ext = temp_base - np.random.uniform(7, 11) + np.random.normal(0, 1)
        elif 6  <= hour <= 12: temp_ext = temp_base - np.random.uniform(0, 4)  + np.random.normal(0, 1)
        elif 13 <= hour <= 16: temp_ext = temp_base + np.random.uniform(0, 3)  + np.random.normal(0, 1)
        else:                  temp_ext = temp_base - np.random.uniform(2, 6)  + np.random.normal(0, 1)
        temp_ext = float(np.clip(temp_ext, 18, 46))

        if month in [6, 7, 8, 9]: humidity = float(np.random.uniform(60, 85))
        else:                       humidity = float(np.random.uniform(15, 45))
        temp_humidity_interaction = temp_ext * (humidity / 100)

        # ── Solaire ───────────────────────────────────────────────────────────
        is_solar_window = 7 <= hour <= 17
        solar_capacity  = 0.0
        if is_solar_window:
            cloud_factor   = max(0.0, 1.0 - humidity / 100 * 0.35)
            base_solar     = 90 if 10 <= hour <= 15 else (50 if hour in [8, 9, 16, 17] else 25)
            solar_capacity = float(max(0.0, base_solar * cloud_factor + np.random.normal(0, 6)))

        # ── SONABEL ───────────────────────────────────────────────────────────
        if   hour == 19:       grid_prob_cut = 0.30
        elif 18 <= hour <= 22: grid_prob_cut = 0.12
        elif temp_ext > 40:    grid_prob_cut = 0.18
        else:                  grid_prob_cut = 0.04
        is_grid_available = 0 if np.random.random() < grid_prob_cut else 1

        # ── Source et prix ────────────────────────────────────────────────────
        if solar_capacity > 20:
            current_source   = 0
            energy_price_kwh = 0
        elif is_grid_available:
            current_source   = 1
            energy_price_kwh = _tarif(hour)
        else:
            current_source   = 2
            energy_price_kwh = TARIF_GE

        # ── Ramadan ───────────────────────────────────────────────────────────
        ramadan_hour_shift = 0
        if is_ram:
            if   hour in [2, 3, 4]:  ramadan_hour_shift = 1
            elif hour in [18, 19]:   ramadan_hour_shift = 2

        # ── Lags (résolution 15 min) ──────────────────────────────────────────
        lags = _extract_lags_from_history(historical_df, step, predictions)

        features = {
            'hour': hour, 'day_of_week': dow,
            'month': month, 'is_weekend': int(is_weekend),
            'holiday_intensity': holiday_intensity,
            'holiday_window': holiday_window,
            'is_end_of_month': int(is_end_of_month),
            'is_ramadan_period': int(is_ram),
            'ramadan_hour_shift': ramadan_hour_shift,
            'temp_ext': temp_ext, 'humidity': humidity,
            'temp_humidity_interaction': temp_humidity_interaction,
            'current_source': current_source,
            'is_grid_available': is_grid_available,
            'energy_price_kwh': energy_price_kwh,
            'is_solar_window': int(is_solar_window),
            **lags,
        }

        X_pred      = pd.DataFrame([features])[FEATURE_COLS]
        # Modèle prédit une puissance en kW
        energy_kw   = float(max(0.0, model.predict(X_pred)[0]))
        # Énergie consommée sur le créneau de 15 min
        energy_kwh  = round(energy_kw * DT_H, 4)

        # Débit demandé (m³/h)
        flow_est = get_water_consumption_pattern(
            hour, is_ram, holiday_intensity, temp_ext, is_weekend, minute)

        # ── Dynamique bâches intégrée dans l'horizon ──────────────────────────
        # On détermine le pompage prédit pour ce pas et on met à jour l'état
        bache_error    = bache_level_pct - 70.0
        q_target       = float(np.clip(flow_est - bache_error * 2.0, 0, Q_TOTAL_MAX))
        n_pumps_pred   = select_n_pumps_heuristic(q_target, bache_level_pct)
        combo_pred     = get_pump_combination(n_pumps_pred)
        q_pomped_pred  = float(combo_pred['flow_m3h'])
        q_inflow_pred  = float(max(0.0, flow_est * 1.05))
        bache_level_pct = simulate_bache_dynamics(
            bache_level_pct, q_pomped_pred, q_inflow_pred, DT_H)

        predictions.append({
            'datetime':                  slot_time.strftime('%Y-%m-%d %H:%M'),
            'date':                      slot_date_str,
            'hour':                      hour,
            'minute':                    minute,
            'temp_ext_predicted':        round(temp_ext, 1),
            'humidity_predicted':        round(humidity, 1),
            'solar_capacity_predicted':  round(solar_capacity, 2),    # kW
            'grid_status_predicted':     is_grid_available,
            'energy_predicted':          round(energy_kw, 2),          # kW
            'energy_kwh_15min':          energy_kwh,                    # kWh sur le créneau
            'energy_price_kwh':          energy_price_kwh,
            'flow_demand_forecast':      round(flow_est, 2),            # m³/h
            'bache_level_pct':           round(bache_level_pct, 2),     # %
            'n_pumps_predicted':         n_pumps_pred,
            'is_ramadan':                bool(is_ram),
            'ramadan_shift':             ramadan_hour_shift,
            'holiday_intensity':         holiday_intensity,
            'is_weekend':                bool(is_weekend),
            'current_source':            current_source,
        })

    with open(PATH_PREDICTIONS, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    log.info(f"✓ 96 prédictions 15 min sauvegardées : {PATH_PREDICTIONS}")

    _build_module2_interface(predictions, historical_df)
    return predictions


# ============================================================================
# [ORC-1] CYCLE MPC COMPLET
# ============================================================================

def run_mpc_iteration(real_data_entry: dict = None, start_time: datetime = None):
    """
    [ORC-1] Orchestre un cycle MPC complet — appelé toutes les **15 min** par APScheduler.

    Alignement automatique : si start_time n'est pas fourni, le cycle démarre
    sur le prochain créneau 15 min aligné (floor au créneau courant).

    Étape 1 : Enregistrement des mesures réelles (si disponibles).
    Étape 2 : Chargement/entraînement du modèle.
    Étape 3 : Prédictions rolling horizon 96 × 15 min + interface Module 2.

    Entrée réelle attendue (real_data_entry) :
      date, hour, minute, energy (kW), flow (m³/h), bache_level_pct,
      temp_ext, humidity, solar_capacity (kW), is_grid_available,
      current_source, n_pumps_active, energy_price_kwh.
    """
    # Aligner start_time sur la grille 15 min
    if start_time is None:
        now  = datetime.now()
        mins = (now.minute // DT_MIN) * DT_MIN
        start_time = now.replace(minute=mins, second=0, microsecond=0)

    log.info("=" * 70)
    log.info(f"CYCLE MPC 15 MIN — {start_time.strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 70)

    if real_data_entry is not None:
        # S'assurer que le champ 'minute' est présent
        real_data_entry.setdefault('minute', start_time.minute)
        log.info("Étape 1/3 — Enregistrement mesures réelles")
        log_real_time_data(real_data_entry)
    else:
        log.info("Étape 1/3 — Aucune mesure réelle (cycle de prédiction seul)")

    log.info("Étape 2/3 — Chargement modèle")
    model = load_or_train_model()

    log.info("Étape 3/3 — Prédictions rolling horizon 96 × 15 min")
    predictions = make_predictions(model, start_time=start_time)

    # Résumé du cycle
    energie_24h = sum(p.get('energy_kwh_15min', p['energy_predicted'] * DT_H)
                      for p in predictions)
    log.info("CYCLE MPC TERMINÉ")
    log.info(f"  Horizon    : {PAS_PAR_JOUR} pas × {DT_MIN} min = 24 h")
    log.info(f"  Énergie 24h: {energie_24h:.1f} kWh (somme des créneaux)")
    log.info(f"  Fichiers   : {PATH_PREDICTIONS} | {PATH_M2_IFACE}")
    log.info("=" * 70)
    return predictions


# ============================================================================
# INITIALISATION DU SYSTÈME
# ============================================================================

def initialize_system():
    """
    Initialise le système Paspanga au premier lancement.
    Génère l'historique simulé, entraîne le modèle, produit les premières prédictions.
    """
    log.info("INITIALISATION SYSTÈME — STATION PASPANGA v1.0")

    if not os.path.exists(PATH_HISTORICAL):
        log.info("Historique absent → génération 365 jours (Paspanga)")
        data = generate_yearly_data(365)
        _save_historical(data)
        with open(PATH_HISTORICAL, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log.info(f"✓ Historique initial : {PATH_HISTORICAL}")
    else:
        log.info(f"✓ Historique existant détecté : {PATH_HISTORICAL}")

    model       = load_or_train_model()
    predictions = make_predictions(model)

    log.info("INITIALISATION TERMINÉE")
    log.info(f"  Station    : Paspanga")
    log.info(f"  Pompes     : 3x90kW + 1x132kW = {P_TOTAL_MAX} kW max")
    log.info(f"  Débit max  : {Q_TOTAL_MAX} m³/h")
    log.info(f"  Bâches     : {N_BACHES}x{V_BACHE_UNIT}m³ = {V_BACHES_TOTAL} m³")
    log.info(f"  Châteaux   : {N_CHATEAUX}x{V_CHATEAU_UNIT}m³ = {V_CHATEAUX_TOTAL} m³ (série C1→C4)")
    log.info(f"  Fichiers   : {PATH_HISTORICAL}, {PATH_PREDICTIONS}, {PATH_M2_IFACE}, {PATH_MODEL}")

    return model, predictions


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    log.info("\n" + "=" * 70)
    log.info("MODULE 1 : PRÉDICTION ÉNERGÉTIQUE — STATION PASPANGA v1.0")
    log.info("=" * 70)

    model, predictions = initialize_system()

    # Exemple appel MPC avec données réelles simulées
    exemple_entree_reelle = {
        'date':              datetime.now().strftime('%Y-%m-%d'),
        'hour':              datetime.now().hour,
        'minute':            (datetime.now().minute // DT_MIN) * DT_MIN,
        'energy':            270.0,         # kW (puissance instantanée)
        'flow':              180.0,          # m³/h
        'bache_level_pct':   65.0,
        'temp_ext':          36.0,
        'humidity':          30.0,
        'solar_capacity':    40.0,           # kW
        'is_grid_available': 1,
        'current_source':    1,
        'n_pumps_active':    2,
        'energy_price_kwh':  TARIF_CREUSE,
    }

    log.info("\nTest run_mpc_iteration() avec données réelles simulées...")
    run_mpc_iteration(real_data_entry=exemple_entree_reelle)

    log.info("\n" + "=" * 70)
    log.info("MODULE 1 PASPANGA — PRÊT POUR APSCHEDULER")
    log.info("=" * 70)
    log.info("Intégration dans app.py :")
    log.info("  from modules.module1_prediction import run_mpc_iteration")
    log.info("  scheduler.add_job(run_mpc_iteration, 'interval', minutes=15)")