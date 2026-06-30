from flask import Flask, render_template, jsonify, request
import json, os, sys
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

# =========================================================
# INIT APP
# =========================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# =========================================================
# STATE SYSTEM (monitoring MPC)
# =========================================================

STATE = {
    "last_mpc_run": None,
    "status": "idle",
    "error": None
}

# =========================================================
# UTILITAIRE JSON
# =========================================================

def _read(fname):
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# =========================================================
# INITIALISATION (UNE SEULE FOIS)
# =========================================================

def initialize_data():
    """
    Initialise uniquement si fichiers absents
    (NE REGENERE PAS à chaque cycle)
    """

    print("=" * 60)
    print(" INITIALISATION PASPANGA MPC SYSTEM")
    print("=" * 60)

    sys.path.insert(0, BASE_DIR)

    from modules.module1_prediction import initialize_system
    from modules.module2_optimization import run_mpc
    from modules.module3_anomalies import detect_anomalies
    from modules.simulator import run_simulator

    # MODULE 1
    if not os.path.exists(os.path.join(DATA_DIR, 'predictions.json')):
        print("[MODULE 1] init...")
        initialize_system()

    # MODULE 2
    if not os.path.exists(os.path.join(DATA_DIR, 'pump_schedule.json')):
        print("[MODULE 2] init MPC...")
        run_mpc()

    # MODULE 3
    if not os.path.exists(os.path.join(DATA_DIR, 'anomalies.json')):
        print("[MODULE 3] init anomalies...")
        detect_anomalies()

    # MODULE 4
    print("[MODULE 4] init simulateur...")
    run_simulator()

    print(" READY\n")

# =========================================================
# MPC CYCLE (TOUTES LES 15 MIN)
# =========================================================

def mpc_cycle():
    """
    ⚠️ NE RECONSTRUIT PAS LES DONNÉES
    exécute uniquement optimisation MPC
    """

    global STATE

    try:
        print("\n==============================")
        print(" MPC CYCLE")
        print("==============================")

        from modules.module2_optimization import run_mpc
        from modules.module3_anomalies import detect_anomalies
        from modules.simulator import run_simulator

        run_mpc()
        run_simulator()
        detect_anomalies()

        STATE["last_mpc_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        STATE["status"] = "success"
        STATE["error"] = None

        print(" MPC DONE")

    except Exception as e:
        STATE["status"] = "error"
        STATE["error"] = str(e)

# =========================================================
# SCHEDULER
# =========================================================

scheduler = BackgroundScheduler()

scheduler.add_job(
    mpc_cycle,
    trigger="interval",
    minutes=15,
    id="mpc_job",
    max_instances=1,
    coalesce=True
)

# =========================================================
# ROUTES FRONT
# =========================================================

@app.route('/')
def home():
    return render_template('index.html')

# =========================================================
# KPI DASHBOARD (IMPORTANT)
# =========================================================

@app.route('/api/kpi')
def api_kpi():
    try:
        metrics = _read('mpc_metrics.json')
        schedule = _read('pump_schedule.json')
        realtime = _read('realtime_state.json')

        if not metrics or not schedule:
            return jsonify({'error': 'missing data'}), 404

        cur = schedule[0]

        return jsonify({
            "cout_total_fcfa": metrics.get("cout_total_fcfa", 0),
            "cout_reference_fcfa": metrics.get("cout_reference_fcfa", 0),
            "economie_fcfa": metrics.get("economie_fcfa", 0),
            "economie_pct": metrics.get("economie_pct", 0),

            "mix_kwh": metrics.get("mix_kwh", {}),
            "part_solaire_pct": metrics.get("part_solaire_pct", 0),

            # AJOUTS IMPORTANTS
            "gasoil_liters": metrics.get("gasoil_liters", 0),
            "co2_kg": metrics.get("co2_kg", 0),

            "chateau_niveaux": realtime.get(
                "chateau_levels_pct_sim",
                metrics.get("chateau_niveaux", [0,0,0,0])
            ),

            "chateau_moyen_pct": realtime.get(
                "chateau_mean_pct_sim",
                metrics.get("chateau_moyen_initial", 0)
            ),

            "bache_niveau_pct": realtime.get(
                "bache_level_pct_sim",
                metrics.get("bache_niveau_initial", 0)
            ),

            "current_pump_label": cur.get("pump_label", "--"),
            "current_n_pumps": cur.get("n_pumps_sim", cur.get("n_pumps", 0)),
            "current_power_kw": cur.get("power_kw_sim", cur.get("power_kw", 0)),
            "current_flow_m3h": cur.get("flow_m3h_sim", cur.get("flow_m3h", 0)),
            "current_source": cur.get("source", "--"),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =========================================================
# DATA ROUTES
# =========================================================

@app.route('/api/predictions')
def api_predictions():
    return jsonify(_read('predictions.json') or [])

@app.route('/api/schedule')
def api_schedule():
    return jsonify(_read('pump_schedule.json') or [])

@app.route('/api/metrics')
def api_metrics():
    return jsonify(_read('mpc_metrics.json') or [])

@app.route('/api/interface')
def api_interface():
    return jsonify(_read('module2_interface.json') or {})

# =========================================================
# NOUVEAUX MODULES RESTAURÉS
# =========================================================

@app.route('/api/anomalies')
def api_anomalies():
    data = _read('anomalies.json')
    if not data:
        return jsonify({
            "status": "module_not_ready",
            "anomalies": []
        })
    # nouveau format : dict avec clé 'anomalies'
    if isinstance(data, dict):
        return jsonify(data)
    # ancien format : liste directe (rétrocompat)
    return jsonify({"anomalies": data})


@app.route('/api/ranking')
def api_ranking():
    data = _read('stations_ranking.json')
    if not data:
        return jsonify({
            "status": "module_not_ready",
            "stations": []
        })
    return jsonify(data)

# =========================================================
# MODULE 4 — SIMULATEUR TEMPS RÉEL
# =========================================================

@app.route('/api/realtime')
def api_realtime():
    data = _read('realtime_state.json')
    if not data:
        return jsonify({"status": "module_not_ready"}), 404
    return jsonify(data)

@app.route('/api/realtime/history')
def api_realtime_history():
    data = _read('realtime_history.json')
    return jsonify(data or [])

@app.route('/api/realtime/events', methods=['GET'])
def api_events_get():
    data = _read('simulator_events.json')
    if not data:
        data = {"fuite_m3h": 0, "pompe_en_panne": False, "capteur_bache_ko": False}
    return jsonify(data)

@app.route('/api/realtime/events', methods=['POST'])
def api_events_set():
    """
    Met à jour simulator_events.json et relance immédiatement le simulateur.
    Body JSON : {"fuite_m3h": 15} ou {"pompe_en_panne": true} etc.
    """
    try:
        import json as _json
        from modules.simulator import run_simulator

        current = _read('simulator_events.json') or \
                  {"fuite_m3h": 0, "pompe_en_panne": False, "capteur_bache_ko": False}

        updates = _json.loads(request.data)
        current.update(updates)

        events_path = os.path.join(DATA_DIR, 'simulator_events.json')
        with open(events_path, 'w') as f:
            _json.dump(current, f, indent=2)

        run_simulator()

        return jsonify({"status": "ok", "events": current})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# =========================================================
# REGENERATE (M2 + M3 uniquement)
# =========================================================

@app.route('/api/regenerate', methods=['POST'])
def api_regenerate():
    try:
        from modules.module2_optimization import run_mpc
        from modules.module3_anomalies import detect_anomalies
        from modules.simulator import run_simulator

        run_mpc()
        run_simulator()
        detect_anomalies()

        STATE["last_mpc_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        STATE["status"] = "success"
        STATE["error"] = None

        return jsonify({"status": "ok"})

    except Exception as e:
        STATE["status"] = "error"
        STATE["error"] = str(e)
        return jsonify({"status": "error", "message": str(e)}), 500

# =========================================================
# SYSTEM STATUS
# =========================================================

@app.route('/api/system')
def system_status():
    return jsonify(STATE)

# =========================================================
# INIT + START
# =========================================================

if __name__ == '__main__':

    initialize_data()

    scheduler.start()

    print("\n====================================")
    print(" ONEA PASPANGA MPC SYSTEM")
    print(" MPC auto chaque 15 minutes")
    print("====================================")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )