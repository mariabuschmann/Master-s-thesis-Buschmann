import subprocess
import time

CONTAINER_NAME = "neuer-gripper"
IMAGE_NAME = "weiss_gripper_image"

def run_command(command):
    """Hilfsfunktion zum Ausführen von Shell-Befehlen."""
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Fehler: {e}")

def start_docker_container():
    """Startet den Docker-Container, falls er nicht läuft."""
    print("Überprüfe, ob der Docker-Container läuft...")

    # Prüfen, ob der Container läuft
    check_running = f"docker ps --filter 'name={CONTAINER_NAME}' --format '{{{{.Names}}}}'"
    result = subprocess.run(check_running, shell=True, capture_output=True, text=True)
    
    if CONTAINER_NAME in result.stdout.strip():
        print(f"Container {CONTAINER_NAME} läuft bereits.")
        return

    print(f"Container {CONTAINER_NAME} läuft nicht. Starte ihn...")

    # Falls der Container existiert, aber gestoppt ist → löschen
    run_command(f"docker rm -f {CONTAINER_NAME}")

    # Container starten
    start_command = (
        f"docker run -dit --name {CONTAINER_NAME} --device=/dev/ttyACM0 --group-add dialout --network host {IMAGE_NAME}"
    )
    run_command(start_command)

    print(f"Docker-Container {CONTAINER_NAME} wurde gestartet.")
    time.sleep(5)  # Warte kurz, bis der Container bereit ist

def make_script_executable():
    """Macht `gripper_close.py` innerhalb des Containers ausführbar."""
    print("Setze Berechtigungen für gripper_close.py im Container...")
    chmod_command = (
        f'docker exec -it {CONTAINER_NAME} bash -c "chmod +x /root/catkin_ws/src/weiss_gripper_ieg76/src/gripper_close.py"'
    )
    run_command(chmod_command)

def run_gripper_command(position=80, force=25):
    """Führt das Gripper-Skript im Docker-Container aus."""
    print(f"Führe Gripper-Kommando aus: Position={position}, Force={force}...")

    gripper_command = (
        f'docker exec -it {CONTAINER_NAME} bash -c "source /root/catkin_ws/devel/setup.bash && '
        f'rosrun weiss_gripper_ieg76 gripper_close.py --position {position} --force {force}"'
    )
    run_command(gripper_command)
    print("Gripper-Kommando erfolgreich ausgeführt.")

def main():
    """Hauptfunktion."""
    start_docker_container()
    make_script_executable()
    run_gripper_command(position=50, force=25)

if __name__ == "__main__":
    main()