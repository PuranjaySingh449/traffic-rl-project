import traci
import sumolib
import csv
import os

SUMO_BINARY = "sumo"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMO_CONFIG = os.path.join(BASE_DIR, "..", "sumo", "config.sumocfg")
OUTPUT_CSV = os.path.join(BASE_DIR, "..", "logs", "baseline_metrics.csv")


def run():
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])

    emergency_depart = None
    emergency_arrive = None

    total_waiting_time = 0.0
    normal_vehicle_count = 0
    total_queue = 0
    steps = 0
    arrived_vehicles = 0

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        steps += 1

        vehicle_ids = traci.vehicle.getIDList()

        for vid in vehicle_ids:
            vtype = traci.vehicle.getTypeID(vid)

            # Emergency vehicle timing
            if vtype == "emergency":
                if emergency_depart is None:
                    emergency_depart = traci.simulation.getTime()

            # Normal vehicle waiting time
            if vtype == "normal":
                total_waiting_time += traci.vehicle.getWaitingTime(vid)
                normal_vehicle_count += 1

        # Queue length (per lane)
        for lane_id in traci.lane.getIDList():
            total_queue += traci.lane.getLastStepHaltingNumber(lane_id)

        arrived_vehicles += traci.simulation.getArrivedNumber()

    # Emergency arrival time
    if emergency_depart is not None:
        emergency_arrive = traci.simulation.getTime()
        emergency_travel_time = emergency_arrive - emergency_depart
    else:
        emergency_travel_time = None

    avg_waiting_time = (
        total_waiting_time / normal_vehicle_count
        if normal_vehicle_count > 0 else 0
    )

    avg_queue_length = total_queue / steps if steps > 0 else 0

    traci.close()

    # Save results
    os.makedirs("../logs", exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "emergency_travel_time",
            "avg_normal_waiting_time",
            "avg_queue_length",
            "throughput"
        ])
        writer.writerow([
            emergency_travel_time,
            avg_waiting_time,
            avg_queue_length,
            arrived_vehicles
        ])

    print("Baseline simulation complete.")
    print(f"Emergency travel time: {emergency_travel_time:.2f}s")
    print(f"Avg normal waiting time: {avg_waiting_time:.2f}s")
    print(f"Avg queue length: {avg_queue_length:.2f}")
    print(f"Throughput: {arrived_vehicles}")

if __name__ == "__main__":
    run()
