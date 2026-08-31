# Kaggle file & column descriptions

The Kaggle CLI validates `resources` in dataset-metadata.json but never
uploads it (`body.files` is hardcoded empty in `dataset_create_version`),
so these have to be entered once in the UI:

  Dataset -> Data tab -> pick the file -> pencil icon.
  Columns have their own pencil icons below; look for a bulk
  "Describe all columns" view to do them in one form.

They persist across versions, so this is a one-time task.

---

## File description

One row per car variant: price, engine, transmission, fuel, mileage and features.

## Column descriptions

**Brand**
Manufacturer, e.g. Maruti Suzuki, Tata, BMW.

**Car**
Model name, e.g. Nexon, Carens Clavis, DB11.

**Variant**
Trim level within the model, e.g. Creative Plus, HTX, Boxster GTS.

**Description**
Generated summary of the variant's fuel, transmission, price, engine, mileage and notable features.

**Price**
Price in Indian rupees. Ranges are stored as their midpoint. Read alongside Price_Type.

**Price_Type**
Whether Price is 'ex-showroom' or 'on-road'. These are NOT comparable -- on-road includes registration and insurance and runs 15-25% higher. Filter on this before any budget comparison.

**Discontinued**
'Yes' if the model is superseded or no longer sold. Discontinued cars usually have no listed price.

**Fuel Type**
Petrol, Diesel, CNG, Electric, LPG or Hybrid.

**Mileage**
Fuel efficiency in kmpl. Often blank for electric cars, which quote range instead.

**Transmission**
Gearbox description, e.g. 'Manual - 5 Gears', 'Automatic (DCT) - 7 Gears'.

**Engine**
Raw engine string as published, e.g. '1199 cc, 3 Cylinders Inline, 4 Valves/Cylinder, DOHC'. Split into the five columns that follow.

**Displacement (cc)**
Engine displacement in cc, parsed from Engine.

**Cylinders**
Number of cylinders, parsed from Engine.

**Cylinder Layout**
Cylinder arrangement: Inline, V, W, Flat.

**Valves per Cylinder**
Valves per cylinder, parsed from Engine.

**Valve Train**
Valve train type: DOHC, SOHC, OHV.

**Cluster Type**
Instrument cluster technology: Analogue, Digital, Analogue - Digital, TFT.

**Cluster Size**
Cluster screen size where stated, e.g. '12.3-inch'. Most cars do not publish this.

**Heads Up Display**
Whether a HUD is fitted: Yes, Optional or no.

**Tachometer**
Tachometer type: Analogue or Digital.

**Display**
Infotainment display description.

**Sunroof / Moonroof**
Yes if a sunroof or moonroof is listed.

**Dashcam**
Yes if a dashcam is listed.

**Rear AC**
Yes if rear air conditioning is listed.

**Central Locking**
Yes if central locking is listed.

**Cruise Control**
Yes if cruise control is listed.

**Hill Hold Control**
Yes if hill hold control is listed.

**Ventilated Seats**
Yes if ventilated seats are listed.

**Wireless Charger**
Yes if wireless phone charging is listed.

**Adjustable ORVMs**
Yes if adjustable outside mirrors are listed.

**Integrated (in-dash) Music System**
Yes if an in-dash music or infotainment system is listed.

**Speakers**
Yes if speakers are listed.

**Instrument Cluster**
Raw instrument cluster feature list, source for the Cluster columns above.

**Combined Description**
All fields concatenated into one text blob, intended for embedding / semantic search.

**Meta_Description**
The site's original SEO meta description. Mostly boilerplate; kept for reference.
