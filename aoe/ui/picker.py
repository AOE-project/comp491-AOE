import cv2
import json

# Configuration
image_path = 'koc_harita.png' 
output_file = 'campus_data.json'

data = {
    "units": "pixels",
    "buildings": []
}

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Prompt user for building details via terminal
        print(f"\nSelected Coordinate: [{x}, {y}]")
        building_name = input("Enter building/point name (e.g., ENG, SNA, Library): ")
        demand = input(f"Enter estimated demand (student count) for {building_name}: ")
        
        # Add entry to data list
        entry = {
            "name": building_name,
            "x": x,
            "y": y,
            "demand": int(demand) if demand.isdigit() else 100
        }
        data["buildings"].append(entry)
        
        # Visualize on map
        cv2.circle(img, (x, y), 7, (0, 255, 0), -1)
        cv2.putText(img, building_name, (x + 10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Data Collection Panel', img)
        print(f"✅ {building_name} saved. You can continue...")

# Load image
img = cv2.imread(image_path)
if img is None:
    print("ERROR: Image not found! Please check the file path.")
else:
    print("--- KOÇ UNIVERSITY DATA COLLECTION TOOL ---")
    print("1. Click on a building on the map.")
    print("2. Enter the name and demand in the terminal.")
    print("3. Press any key to SAVE and EXIT when finished.")
    
    cv2.imshow('Data Collection Panel', img)
    cv2.setMouseCallback('Data Collection Panel', click_event)
    cv2.waitKey(0)
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"\n🚀 All data has been successfully saved to '{output_file}'!")
    cv2.destroyAllWindows()