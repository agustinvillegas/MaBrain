"""Massive synthetic English dataset generator — data-driven, combinatorial."""
import json, random, os, sys
sys.path.insert(0, os.path.dirname(__file__))

from expand_dataset_en import (
    IS_A, NEXT, HAS, HAS_PROPERTY, FUNCTION, CAUSE, PART_OF,
    INSTRUMENT, LOCATED_IN, OPPOSITE, SIMILAR_TO, MADE_OF,
    USED_FOR,
)

random.seed(42)

# ── Word lists (normalized: lowercase, no punctuation) ───────────
ANIMALS     = ["dog","cat","wolf","fox","lion","tiger","horse","cow","sheep","pig",
               "goat","deer","bear","rabbit","squirrel","mouse","rat","bat",
               "whale","dolphin","seal","elephant","giraffe","zebra","camel",
               "monkey","gorilla","panda","kangaroo","koala","moose","bison"]
BIRDS       = ["eagle","hawk","owl","raven","crow","pigeon","sparrow","parrot",
               "swan","duck","goose","penguin","peacock","turkey","chicken",
               "rooster","hen","ostrich","flamingo"]
FISH        = ["salmon","tuna","trout","cod","carp","goldfish","shark","eel"]
REPTILES    = ["snake","lizard","turtle","crocodile","iguana","chameleon",
               "tortoise","gecko","python","cobra"]
INSECTS     = ["ant","bee","wasp","fly","mosquito","butterfly","moth",
               "ladybug","dragonfly","grasshopper","spider","scorpion"]
PLANTS      = ["oak","maple","pine","birch","willow","cedar","palm","cactus",
               "rose","daisy","tulip","sunflower","orchid","lily","fern","moss"]
VEGETABLES  = ["carrot","broccoli","spinach","lettuce","cabbage","pepper",
               "tomato","cucumber","onion","garlic","potato","corn","peas","beans"]
FRUITS      = ["apple","banana","orange","grape","strawberry","blueberry","cherry",
               "peach","plum","pear","watermelon","mango","kiwi","lemon","lime"]
FOODS       = ["bread","rice","pasta","cheese","butter","milk","yogurt","egg",
               "meat","chicken_meat","soup","salad","pizza","burger","sandwich"]
BEVERAGES   = ["water","juice","soda","tea","coffee","lemonade","smoothie"]
TOOLS       = ["hammer","screwdriver","wrench","pliers","saw","drill","chisel",
               "axe","knife","scissors","shovel","rake","ladder","rope","lock","key"]
PROFESSIONS = ["doctor","nurse","teacher","engineer","pilot","police","firefighter",
               "judge","lawyer","scientist","artist","writer","musician","chef",
               "farmer","carpenter","plumber","programmer"]
BODY_PARTS  = ["head","ear","eye","nose","mouth","tongue","tooth","lip","chin",
               "neck","shoulder","arm","elbow","wrist","hand","finger","thumb",
               "chest","back","stomach","hip","leg","knee","ankle","foot","toe",
               "heart","lung","liver","kidney","brain","skin","bone","muscle"]
EMOTIONS    = ["joy","happiness","love","hope","peace","gratitude","excitement",
               "sadness","anger","fear","anxiety","loneliness","disgust","surprise"]
SPORTS      = ["soccer","basketball","baseball","tennis","volleyball","hockey",
               "golf","swimming","cycling","boxing","judo","karate","fencing",
               "skiing","surfing","diving","climbing","yoga"]
GEOGRAPHY   = ["mountain","river","lake","ocean","sea","desert","forest","jungle",
               "island","valley","hill","plain","cave","waterfall","volcano"]
WEATHER     = ["rain","snow","wind","storm","thunder","cloud","sun","ice","fog"]
TECH        = ["computer","laptop","phone","tablet","tv","keyboard","mouse",
               "internet","software","hardware","database","server","network",
               "app","website","algorithm","robot","sensor","battery"]
VEHICLES    = ["car","truck","bus","train","plane","boat","bicycle","motorcycle",
               "rocket","ship"]
COLORS      = ["red","blue","green","yellow","purple","orange","black","white",
               "gray","brown","pink","gold"]
SHAPES      = ["circle","square","triangle","rectangle","oval","diamond","star"]
ABSTRACT    = ["time","space","energy","truth","beauty","freedom","justice",
               "knowledge","wisdom","power","health","peace_abs"]
MUSIC       = ["guitar","piano","drums","violin","flute","trumpet","saxophone","cello"]
ACTIONS     = ["pump","breathe","filter","think","see","hear","smell","taste","feel",
               "digest","contract","support","teach","heal","cook","paint","write",
               "play_music","fly","pound","turn","cut","slice","unlock","compute",
               "call","shine","fall","blow","burn","flow","tick","drive","bark",
               "meow","neigh","moo","roar","howl","quack","cluck","fly_bird","swim",
               "slither","hop","jump","grow","erode","freeze","melt","damage","sweat",
               "paint_art","code","see_vision","hear_audio","smell_odor",
               "germinate","photosynthesize","bloom","sprout","reproduce",
               "rescue","defend","attack","communicate","navigate","build",
               "growl","chirp","hoot","sing","peck","hiss","crow","climb","gallop",
               "stalk","pounce","graze","forage","hunt","migrate","hibernate",
               "spin","weave","pollinate","sting","bite",
               "store","retrieve","process","transmit","charge","display","type"]
CLOTHING    = ["shirt","pants","shoes","socks","jacket","coat","hat","scarf",
               "gloves","belt","dress","skirt","sweater","jeans"]
FURNITURE   = ["table","chair","sofa","bed","desk","bookshelf","lamp","mirror"]

# All words pool
ALL_WORDS = set()
for lst in [ANIMALS, BIRDS, FISH, REPTILES, INSECTS, PLANTS, VEGETABLES, FRUITS,
            FOODS, BEVERAGES, TOOLS, PROFESSIONS, BODY_PARTS, EMOTIONS, SPORTS,
            GEOGRAPHY, WEATHER, TECH, VEHICLES, COLORS, SHAPES, ABSTRACT, MUSIC,
            CLOTHING, FURNITURE]:
    ALL_WORDS.update(lst)

# Add taxonomic super-categories
TAXONOMIC_LABELS = [
    "canine","feline","equine","bovine","ovine","suid","rodent","lagomorph",
    "cetacean","pinniped","primate","ursid","elephantid","giraffid","camelid",
    "macropod","marsupial","phascolarctid","deer","bat",
    "raptor","psittacine","waterfowl","spheniscid","phasianid",
    "fish_species","elasmobranch",
    "serpent","lacertilian","chelonian","crocodylian",
    "formicid","apid","vespid","dipteran","culicid","lepidopteran",
    "coleopteran","odonatan","orthopteran","arachnid",
    "carnivore","herbivore","omnivore",
    "mammal","vertebrate","invertebrate","bird","fish","reptile","insect",
    "animal","living_thing",
    "flower","conifer","tree","plant","succulent",
    "root_vegetable","brassica","leafy_green","fruit_vegetable","gourd",
    "bulb_vegetable","tuber","grain","legume","vegetable","produce",
    "pome_fruit","drupe","berry","fruit","food_plant",
    "digital_device","device","vehicle","machine","aircraft","watercraft",
    "cycle","technology",
    "string_instrument","keyboard_instrument","wind_instrument",
    "brass_instrument","percussion","instrument","tool",
    "artifact","professional","body_part","emotion","sport",
    "geography","weather_phenomenon","color","shape","food","beverage",
    "clothing","furniture","abstract_concept",
]
ALL_WORDS.update(TAXONOMIC_LABELS)
ALL_WORDS.update(ACTIONS)

# Auto-add all words from relation pairs
EXTRA_WORDS = set()
# From function_agents
for a, b in {"heart":"pump","lung":"breathe","liver":"filter","kidney":"filter",
        "brain":"think","eye":"see","ear":"hear","nose":"smell","tongue":"taste",
        "skin":"feel","stomach":"digest","muscle":"contract","bone":"support",
        "teacher":"teach","doctor":"heal","chef":"cook","artist":"paint",
        "writer":"write","musician":"play_music","pilot":"fly","programmer":"code",
        "hammer":"pound","screwdriver":"turn","saw":"cut","knife":"slice",
        "key":"unlock","computer":"compute","phone":"call",
        "sun":"shine","rain":"fall","wind":"blow","fire":"burn","water":"flow",
        "clock":"tick","river":"flow","car":"drive","plane":"fly",
        "dog":"bark","cat":"meow","horse":"neigh","cow":"moo","lion":"roar",
        "wolf":"howl","duck":"quack","chicken":"cluck","bird":"fly_bird",
        "fish":"swim","snake":"slither","rabbit":"hop","kangaroo":"jump",
        "eagle":"soar","owl":"hoot","parrot":"talk","frog":"croak",
        "cricket":"chirp","bear":"growl","elephant":"trumpet_call",
        "bicycle":"ride","boat":"sail","rocket":"launch",
        "robot":"compute","sensor":"detect","battery":"charge","solar_panel":"generate",
        "flower":"bloom","tree":"grow","seed":"germinate","plant":"photosynthesize",
        "spider":"spin","bee":"pollinate","butterfly":"flutter",
        "police":"protect","firefighter":"rescue","soldier":"defend",
        "farmer":"cultivate","carpenter":"build","plumber":"repair",
        "judge":"judge","scientist":"research","engineer":"design",
        "moon":"reflect","star":"twinkle",}.items():
    EXTRA_WORDS.add(a); EXTRA_WORDS.add(b)
# From cause_pairs
for a,b in [("fire","heat"),("sun","light"),("rain","growth"),("wind","erosion"),
        ("cold","freeze"),("heat","melt"),("friction","heat"),
        ("storm","damage"),("exercise","sweat"),("study","knowledge"),
        ("cloud","rain"),("volcano","eruption"),("seed","sprout"),
        ("drought","famine"),("flood","damage"),("virus","illness"),
        ("smoke","cough"),("alcohol","intoxication"),("gravity","fall"),
        ("electricity","shock"),("pressure","compression"),("sunburn","pain"),
        ("love","joy"),("hate","anger"),("loss","sadness"),
        ("exercise","strength"),("practice","skill"),("sleep","rest"),
        ("food","energy"),("water","hydration"),("medicine","healing"),
        ("explosion","destruction"),("pollution","disease"),
        ("deforestation","erosion"),("fossil_fuel","pollution"),
        ("training","endurance"),("education","wisdom"),
        ("rain","flood"),("earthquake","destruction"),]:
    EXTRA_WORDS.add(a); EXTRA_WORDS.add(b)
# From part_pairs
for a,b in [("finger","hand"),("hand","arm"),("arm","body"),("toe","foot"),
        ("foot","leg"),("leg","body"),("ear","head"),("eye","head"),
        ("nose","head"),("mouth","head"),("head","body"),("heart","body"),
        ("lung","body"),("liver","body"),("kidney","body"),("brain","head"),
        ("skin","body"),("bone","body"),("muscle","body"),
        ("stomach","body"),("intestine","body"),
        ("leaf","tree"),("branch","tree"),("trunk","tree"),("root","tree"),
        ("petal","flower"),("stem","plant"),("seed","fruit"),
        ("wheel","car"),("door","car"),("engine","car"),("seat","car"),
        ("screen","phone"),("button","phone"),("key","keyboard"),
        ("page","book"),("chapter","book"),("cover","book"),
        ("room","house"),("floor","building"),("wall","building"),
        ("wing","bird"),("beak","bird"),("feather","bird"),("tail","animal"),
        ("claw","animal"),("horn","animal"),("hoof","horse"),
        ("fin","fish"),("scale","fish"),("gill","fish"),
        ("blade","knife"),("handle","tool"),("head","hammer"),
        ("string","guitar"),("pedal","piano"),
        ("lens","camera"),("antenna","radio"),
        ("blade","fan"),
        ("sail","boat"),("oar","boat"),("rudder","boat"),
        ("tire","car"),("steering_wheel","car"),("brake","car"),
        ("saddle","horse"),("bridle","horse"),
        ("shelf","bookcase"),("drawer","desk"),]: EXTRA_WORDS.add(a); EXTRA_WORDS.add(b)
# From loc_pairs
for a,b in [("fish","water"),("bird","sky"),("whale","ocean"),("cactus","desert"),
        ("teacher","school"),("doctor","hospital"),("chef","kitchen"),
        ("car","road"),("plane","airport"),("train","station"),
        ("book","library"),("food","kitchen"),
        ("polar_bear","arctic"),("penguin","antarctic"),("camel","desert"),
        ("monkey","jungle"),("bear","forest"),("eagle","mountain"),
        ("lion","savanna"),("tiger","jungle"),("shark","ocean"),
        ("parrot","tropical_forest"),("koala","eucalyptus_forest"),
        ("plant","pot"),("clothes","closet"),
        ("tools","workshop"),("garage","parking"),
        ("pirate","ship"),("robot","factory"),("sensor","robot_body"),
        ("doctor","clinic"),("teacher","classroom"),("chef","restaurant"),
        ("artist","studio"),("librarian","library"),
        ("sun","sky"),("moon","sky"),("star","space"),
        ("money","bank"),("refrigerator","kitchen_storage"),
        ("fruit","tree"),("flower","garden"),("weed","field"),
        ("iceberg","ocean"),("volcano","island"),]:
    EXTRA_WORDS.add(a); EXTRA_WORDS.add(b)
# From instrument triplets
for a,t,act in [("carpenter","hammer","pound"),("chef","knife","slice"),
        ("writer","pen","write"),("artist","brush","paint"),
        ("farmer","shovel","dig"),("lumberjack","saw","cut"),
        ("mechanic","wrench","tighten"),("programmer","computer","code"),
        ("musician","guitar","play_music"),("doctor","stethoscope","examine"),
        ("carpenter","drill","make_hole"),("painter","roller","paint_wall"),
        ("gardener","rake","gather"),("teacher","chalk","teach"),
        ("student","book","learn"),("fisherman","net","catch_fish"),
        ("artist","pencil","sketch"),("scientist","microscope","observe"),
        ("astronomer","telescope","observe"),("surgeon","scalpel","operate"),
        ("sculptor","chisel","sculpt"),("photographer","camera","photo"),
        ("writer","keyboard","type"),("chef","spatula","flip"),
        ("detective","magnifying_glass","investigate"),("lumberjack","axe","chop"),
        ("soldier","gun","shoot"),("pilot","controls","fly_plane"),
        ("musician","piano","play_piano"),]:
    EXTRA_WORDS.add(a); EXTRA_WORDS.add(t); EXTRA_WORDS.add(act)
# From property_pairs
for a,b in [("rose","red"),("sky","blue"),("grass","green"),("snow","white"),
        ("night","dark"),("sun","bright"),("ice","cold"),("fire","hot"),
        ("lemon","sour"),("sugar","sweet"),("coffee","bitter"),
        ("elephant","big"),("mouse","small"),("cheetah","fast"),
        ("turtle","slow"),("diamond","hard"),("pillow","soft"),
        ("gold","yellow"),("orange_fruit","orange_color"),("banana","yellow"),
        ("apple","red"),("grape","purple"),
        ("iron","strong"),("glass","fragile"),("rubber","elastic"),
        ("lead","heavy"),("feather","light_weight"),("steel","hard"),
        ("tiger","striped"),("wolf","wild"),("dog","domestic"),
        ("silk","smooth"),("sandpaper","rough"),("ice","slippery"),
        ("cactus","spiky"),("mountain","tall"),("ocean","deep"),]:
    EXTRA_WORDS.add(a); EXTRA_WORDS.add(b)
# From opp_pairs
for a,b in [("hot","cold"),("big","small"),("fast","slow"),("high","low"),
        ("light","dark"),("day","night"),("wet","dry"),("clean","dirty"),
        ("rich","poor"),("happy","sad"),("love","hate"),("life","death"),
        ("new","old"),("strong","weak"),("hard","soft"),("thick","thin"),
        ("open","close"),("start","finish"),("enter","exit"),
        ("buy","sell"),("give","take"),("create","destroy"),
        ("full","empty"),("heavy","light_weight"),("sharp","dull"),
        ("sweet","sour"),("loud","quiet"),("rough","smooth"),
        ("bright","dim"),("wide","narrow"),("deep","shallow"),
        ("kind","cruel"),("brave","cowardly"),("generous","stingy"),
        ("victory","defeat"),("success","failure"),("health","illness"),
        ("peace","war"),("order","chaos"),("freedom","captivity"),
        ("arrive","depart"),("awake","asleep"),("contract","expand"),
        ("ascend","descend"),("import","export"),("include","exclude"),]:
    EXTRA_WORDS.add(a); EXTRA_WORDS.add(b)

# Extra vocabulary from expanded structural pairs
EXTRA_VOCAB = {
    # body / anatomy
    "artery","vein","capillary","pancreas","thyroid","spine","rib","jaw",
    "pupil","eyelid","eyebrow","face","nerve","neuron","lobe","cortex",
    "circulatory_system","nervous_system","immune_system","intestine",
    "skull","palm","thumb","ankle","shoulder","hip","spine","rib",
    "fingernail","knuckle","shin","thigh","calf","bicep","tricep",
    # professions
    "dentist","veterinarian","electrician","mechanic","architect",
    "librarian","accountant","photographer","astronaut","sailor",
    "barber","tailor","janitor","cashier","banker","driver",
    "athlete","golfer","tennis_player","baseball_player","boxer",
    "fencer","sculptor","potter","hunter","pirate","king","queen",
    "monk","nun","professor","actor","singer","worker",
    # tools / objects
    "file","plane","level","tape","nail","plow","tractor","scythe",
    "hoe","watering_can","pruner","hose","jack","ratchet","socket",
    "welder","grease","cad","stylus","syringe","thermometer",
    "forceps","suture","xray","beaker","pipette","centrifuge",
    "spectrometer","bunsen_burner","tripod","flash","mallet",
    "kiln","whiteboard","projector","calculator","rod","hook",
    "bait","bow","trap","handcuffs","badge","flashlight","notebook",
    "broom","mop","vacuum","rag","comb","razor","needle","thread",
    "measuring_tape","ball","racket","glove","club","foil",
    "printer","scanner","router","cpu","ram","hard_drive","monitor",
    "speaker","microphone","charger","chip",
    "roller","stethoscope","scalpel","magnifying_glass",
    # natural phenomena / geography
    "humidity","lightning","tornado","hurricane","tsunami","avalanche",
    "mudslide","glacier","aurora","rainbow","coast","beach","pond",
    "meadow","harbor","land","continent","peninsula","observatory",
    "gym","laboratory","monastery","convent","palace","castle",
    "launch_pad","helipad","port","bus_stop","bike_lane","stage",
    "theater","court","barracks","basement","aquarium",
    # food / kitchen
    "blender","grater","peeler","colander","stove","oven",
    # new actions
    "oxygenate","detoxify","circulate","absorb","chew","incise",
    "carry_blood","return_blood","produce_insulin","regulate_metabolism",
    "support_body","protect_organs","bend_arm",
    "drill_tooth","treat_animal","design_building","repair_car",
    "fix_pipe","organize_books","enforce_law","grow_food",
    "capture_image","explore_space","bore","carve","smooth",
    "shape","grip","trim","reach_high","secure","bind","flip",
    "beat","flatten","listen_heart","magnify","enlarge","photograph",
    "write_board","incise","store_charge","cover","boom","strike",
    "spin_web","build_dam","chug","transport","speed","dive","haul",
    "print","scan","route_traffic","serve_data","store_data",
    "convert_light","rotate","combust","spin_generator",
    "produce_electricity","absorb_water","ripen","store_water",
    "spore","cover_ground","cover_soil",
    # physical / abstract descriptors
    "light_weight","salty","viscous","fluid","malleable","brittle",
    "transparent","opaque","reflective",
    "domestic","wild","noisy","silent","peaceful","calm",
    # spaces / places
    "courtroom","classroom","bedroom","bathroom","living_room",
    "helipad","bus_stop","parking_lot","sidewalk",
    "space_station","observatory","laboratory",
    # instruments / materials
    "clay","concrete","steel","iron","cotton","silk","wool",
    "marble","stone","brick","glass","paper","wood","leather",
    "plastic","rubber","metal",
    # miscellaneous domains
    "colorful","spotted","striped","thorny","spiky","shiny","furry",
    "feathered","scaly",
    "slipperiness","weight","gravity","magnetism","electric_current",
    "sound_vibration","light_reflection","oxidation","corrosion",
    # more body-related
    "circulatory_system","skeletal_system","muscular_system",
    "digestive_system","respiratory_system","nervous_system",
    # CAUSE missing words
    "acceleration","achievement","acid","aging","allergen","allergy",
    "art","attraction","automation","bacteria","bleaching","bleeding",
    "burial","climate_change","combustion","computation","conflict",
    "connectivity","contraction","crime","crop_failure","danger","darkness",
    "data","dehydration","depletion","disappointment","discovery",
    "efficiency","encryption","endorphin","evaporation","expansion",
    "experiment","fever","force","frost","habitat_loss","hearing","hunger",
    "inequality","infection","inflammation","information","inspiration",
    "insult","isolation","kindness","landslide","learning","mastery",
    "meditation","moisture","mold","music","nuclear_fission","observation",
    "overfishing","pathogen","performance","photosynthesis","plant_growth",
    "poison","pollen","poverty","pride","reading","repetition","rust",
    "security","sneeze","solar_radiation","static_electricity","stress",
    "suffering","sunlight","thirst","threat","thunderstorm","understanding",
    "unrest","urbanization","vision","vitamin_d","wave","weakness","wrinkle",
    # INSTRUMENT missing words
    "aim","analyze_light","anchor","argue","attract_fish","bake","beaver",
    "binoculars","biologist","bleat","blend","bray","buzz","calculate",
    "camouflage","canvas","capture","care","carry","catch","caw","chain",
    "charcoal","chemist","clean_carpet","clean_floor","click","compass",
    "control_speed","coo","coyote","cut_fabric","cut_hair","designer",
    "document","donkey","dove","drain","draw","drum","easel","eat_wood",
    "echolocate","erupt","examine_animal","examine_cell","examine_tooth",
    "extinguish","fight","filter_blood","fire_pottery","firefly","float",
    "generator","glow","gobble","grasp","harvest","helicopter","hit",
    "hit_ball","hit_pitch","hold_canvas","hover","identify","illuminate",
    "inject","join","lift","loosen","lubricate","measure","measure_body",
    "measure_liquid","measure_temp","mix","moor","motor","move","oink",
    "paint_on","peel","play_cello","play_flute","play_saxophone",
    "play_trumpet","play_violin","plow_field","present","punch","radar",
    "radiate","record","restrain","retreat","ribbit","rifle","rolling_pin",
    "screech","search","sentence","separate","serve","sew","shake","shave",
    "shoot_arrow","shred","solar_cell","spin_web","spread","squeak",
    "stabilize","steer_boat","steer_plane","stitch","stitch_wound","style_hair",
    "submarine","sweep","swing","teach_from","termite","throttle","throw",
    "throw_pottery","thrust","tighten_bolt","till","transmit_signal",
    "turbine","typewriter","vine","water_plants","weld","whisk","wipe",
    "wire","write_on","yelp","yoke","zoom",
    # OPPOSITE missing words
    "above","above_ground","accelerate","accept","admit","advance",
    "adversary","affluent","airy","alert","alive","allow","ally","anarchy",
    "ancient","approach","arid","assemble","attach","attract","aware",
    "bankrupt","begin","below","bland","blunt","boiling","boisterous",
    "borrow","bottom","break","broad","bumpy","chaotic","cheerful",
    "civilization","coarse","come","commence","conclude","connect",
    "conscious","construct","contaminated","cool","crowded","damp","dawn",
    "dead","deafening","deceased","deceitful","decelerate","decline",
    "deflate","deform","dehydrated","demand","demolish","dense","deny",
    "dependence","desert_dry","deserted","despairing","destitute","detach",
    "detune","disable","disassemble","disaster","discharge","disconnect",
    "discord","dismantle","down","drowsy","dusk","early","earn","elevated",
    "enable","end","enemy","enormous","ethical","evening","evil","faint",
    "fasten","fat","feeble","filled","fine","firm","fit","flat","flexible",
    "flimsy","forbid","form","freezing","friend","future","gain","gentle",
    "gloomy","go","good","gradual","harmony","hasty","healed","healthy",
    "hollow","honest","hopeful","huge","humid","illuminated","immoral",
    "impatient","impoverished","independence","infected","inflate","jagged",
    "joyful","keen","large","late","leisurely","lend","liberty","limited",
    "liquid","living","long","loosen","lose","loyal","make","massive",
    "messy","mild","miniature","minute","miserable","modern","moist","moral",
    "morning","murky","oblivious","offer","oppression","optimistic","orderly",
    "packed","parched","past","patient","permit","pessimistic","pointed",
    "ponderous","populated","powerful","profit","profound","prohibit",
    "prosperous","pull","pure","push","quick","radiant","rapid","raucous",
    "reject","repel","retail","retreat","rigid","rise","robust","rugged",
    "save","savory","shadowy","short","shut","sick","skinny","sleek",
    "sluggish","soaked","solid","sovereignty","sparse","spend","spicy",
    "sterile","sturdy","subjugation","sunken","superficial","supply","swift",
    "tender","tidy","tie","tiny","top","tough","traitorous","triumph",
    "tune","unconscious","underground","unethical","uneven","unfit",
    "uninhabited","untie","unwell","up","vacant","vast","violent","virtuous",
    "warm","waste","wealthy","weightless","well","wholesale","wicked","win",
    "withdraw","wounded","young",
    # PART_OF / LOCATED_IN missing
    "beak","claw","feather","fin","gill","hoof","horn","scale","tail",
    "wing","branch","leaf","root","trunk","petal","seed","stem",
    "blade","handle","string","pedal","lens","antenna","button",
    "screen","wheel","engine","seat","door",
    "sail","oar","rudder","tire","steering_wheel","brake",
    "chapter","cover","page","room","floor","wall",
    "keyboard","piano","guitar","camera","radio","fan","laptop",
    "boat","car","phone","book","house","building",
}
EXTRA_WORDS.update(EXTRA_VOCAB)

ALL_WORDS.update(EXTRA_WORDS)
print(f"Vocabulary + extra: {len(ALL_WORDS)} words")

# ── IS_A hierarchy (word → direct category) ───────────────────
CATEGORY_MAP = {
    # animals
    "dog":"canine","wolf":"canine","fox":"canine",
    "cat":"feline","lion":"feline","tiger":"feline",
    "horse":"equine","zebra":"equine","donkey":"equine",
    "cow":"bovine","sheep":"ovine","goat":"ovine","pig":"suid",
    "rabbit":"lagomorph","squirrel":"rodent","mouse":"rodent","rat":"rodent","bat":"bat",
    "whale":"cetacean","dolphin":"cetacean","seal":"pinniped",
    "elephant":"elephantid","giraffe":"giraffid","camel":"camelid",
    "monkey":"primate","gorilla":"primate","panda":"ursid","bear":"ursid",
    "kangaroo":"macropod","koala":"phascolarctid","moose":"deer","bison":"bovine",
    "eagle":"raptor","hawk":"raptor","owl":"raptor",
    "parrot":"psittacine","swan":"waterfowl","duck":"waterfowl","goose":"waterfowl",
    "penguin":"spheniscid","peacock":"phasianid","turkey":"phasianid",
    "chicken":"phasianid","rooster":"phasianid","hen":"phasianid",
    "salmon":"fish_species","tuna":"fish_species","trout":"fish_species",
    "cod":"fish_species","carp":"fish_species","goldfish":"fish_species",
    "shark":"elasmobranch","eel":"fish_species",
    "snake":"serpent","lizard":"lacertilian","turtle":"chelonian",
    "crocodile":"crocodylian","iguana":"lacertilian","chameleon":"lacertilian",
    "tortoise":"chelonian","gecko":"lacertilian","python":"serpent","cobra":"serpent",
    "ant":"formicid","bee":"apid","wasp":"vespid","fly":"dipteran",
    "mosquito":"culicid","butterfly":"lepidopteran","moth":"lepidopteran",
    "ladybug":"coleopteran","dragonfly":"odonatan","grasshopper":"orthopteran",
    "spider":"arachnid","scorpion":"arachnid",
    # plants
    "rose":"flower","daisy":"flower","tulip":"flower","sunflower":"flower",
    "orchid":"flower","lily":"flower","oak":"tree","maple":"tree","pine":"conifer",
    "birch":"tree","willow":"tree","cedar":"conifer","palm":"tree","cactus":"succulent",
    "carrot":"root_vegetable","broccoli":"brassica","lettuce":"leafy_green",
    "cabbage":"brassica","pepper":"fruit_vegetable","tomato":"fruit_vegetable",
    "cucumber":"gourd","onion":"bulb_vegetable","garlic":"bulb_vegetable",
    "potato":"tuber","corn":"grain","peas":"legume","beans":"legume",
    "apple":"pome_fruit","pear":"pome_fruit","peach":"drupe","plum":"drupe",
    "cherry":"drupe","grape":"berry","strawberry":"berry","blueberry":"berry",
    # technology
    "computer":"digital_device","laptop":"digital_device","phone":"digital_device",
    "tablet":"digital_device","tv":"digital_device",
    # vehicles
    "car":"vehicle","truck":"vehicle","bus":"vehicle","train":"vehicle",
    "plane":"aircraft","boat":"watercraft","bicycle":"cycle","motorcycle":"cycle",
    # music
    "guitar":"string_instrument","piano":"keyboard_instrument",
    "violin":"string_instrument","cello":"string_instrument",
    "flute":"wind_instrument","trumpet":"brass_instrument",
    "saxophone":"brass_instrument","drums":"percussion",
}

SUPER_CATEGORIES = {
    "canine":"carnivore","feline":"carnivore",
    "carnivore":"mammal","herbivore":"mammal","omnivore":"mammal",
    "equine":"herbivore","bovine":"herbivore","ovine":"herbivore",
    "suid":"omnivore","rodent":"mammal","lagomorph":"herbivore",
    "cetacean":"mammal","pinniped":"mammal","primate":"mammal","ursid":"mammal",
    "elephantid":"mammal","giraffid":"herbivore","camelid":"herbivore",
    "macropod":"marsupial","phascolarctid":"marsupial","deer":"herbivore","bat":"mammal",
    "raptor":"bird","psittacine":"bird","waterfowl":"bird","spheniscid":"bird",
    "phasianid":"bird",
    "fish_species":"fish","elasmobranch":"fish",
    "serpent":"reptile","lacertilian":"reptile","chelonian":"reptile",
    "crocodylian":"reptile",
    "formicid":"insect","apid":"insect","vespid":"insect","dipteran":"insect",
    "culicid":"insect","lepidopteran":"insect","coleopteran":"insect",
    "odonatan":"insect","orthopteran":"insect","arachnid":"arachnid",
    "mammal":"vertebrate","bird":"vertebrate","fish":"vertebrate",
    "reptile":"vertebrate","insect":"invertebrate","arachnid":"invertebrate",
    "vertebrate":"animal","invertebrate":"animal",
    "flower":"plant","conifer":"tree","tree":"plant","succulent":"plant",
    "root_vegetable":"vegetable","brassica":"vegetable","leafy_green":"vegetable",
    "fruit_vegetable":"vegetable","gourd":"vegetable","bulb_vegetable":"vegetable",
    "tuber":"vegetable","grain":"vegetable","legume":"vegetable",
    "pome_fruit":"fruit","drupe":"fruit","berry":"fruit",
    "fruit":"produce","vegetable":"produce",
    "digital_device":"device","vehicle":"machine","aircraft":"vehicle",
    "watercraft":"vehicle","cycle":"vehicle","machine":"technology",
    "string_instrument":"instrument","keyboard_instrument":"instrument",
    "wind_instrument":"instrument","brass_instrument":"instrument",
    "percussion":"instrument","instrument":"tool",
    "animal":"living_thing","plant":"living_thing",
}

def isa_chain(word, direct_cat, super_cats):
    """Generate IS_A: word → cat → super_cat → ... until root."""
    chain = [word]
    cat = direct_cat
    while cat:
        chain.append(cat)
        cat = super_cats.get(cat)
    return [IS_A] * (len(chain) - 1), chain


# ── Generate sequences ─────────────────────────────────────────
def main(output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    all_seqs = []

    # 1. IS_A chains for every classified word
    for word, cat in CATEGORY_MAP.items():
        if word not in ALL_WORDS:
            continue
        rels, chain = isa_chain(word, cat, SUPER_CATEGORIES)
        if len(chain) >= 2:
            all_seqs.append({"sequence": chain, "relations": rels})

    # 2. FUNCTION pairs
    function_agents = {
        # body functions
        "heart":"pump","lung":"breathe","liver":"filter","kidney":"filter",
        "brain":"think","eye":"see","ear":"hear","nose":"smell","tongue":"taste",
        "skin":"feel","stomach":"digest","muscle":"contract","bone":"support",
        "artery":"carry_blood","vein":"return_blood","intestine":"absorb",
        "pancreas":"produce_insulin","thyroid":"regulate_metabolism",
        "spine":"support_body","rib":"protect_organs","knee":"bend",
        "elbow":"bend_arm","jaw":"chew","tooth":"bite","tongue":"taste",
        "lung":"oxygenate","liver":"detoxify","kidney":"filter_blood",
        "immune_system":"defend","nerve":"transmit_signal",
        # professions
        "teacher":"teach","doctor":"heal","chef":"cook","artist":"paint",
        "writer":"write","musician":"play_music","pilot":"fly","programmer":"code",
        "nurse":"care","dentist":"drill_tooth","veterinarian":"treat_animal",
        "architect":"design_building","electrician":"wire","mechanic":"repair_car",
        "plumber":"fix_pipe","gardener":"plant","librarian":"organize_books",
        "accountant":"calculate","soldier":"fight","police":"enforce_law",
        "judge":"sentence","lawyer":"argue","scientist":"experiment",
        "engineer":"build","firefighter":"extinguish","farmer":"grow_food",
        "photographer":"capture_image","astronaut":"explore_space",
        # tools
        "hammer":"pound","screwdriver":"turn","saw":"cut","knife":"slice",
        "key":"unlock","computer":"compute","phone":"call","drill":"bore",
        "chisel":"carve","plane":"smooth","file":"shape","wrench":"tighten",
        "pliers":"grip","scissors":"trim","axe":"chop","shovel":"dig",
        "rake":"gather","ladder":"reach_high","lock":"secure","chain":"bind",
        "spatula":"flip","whisk":"beat","rolling_pin":"flatten",
        "stethoscope":"listen_heart","microscope":"magnify","telescope":"enlarge",
        "camera":"photograph","pencil":"sketch","brush":"paint","pen":"write",
        "chalk":"write_board","keyboard":"type","mouse":"click",
        "magnifying_glass":"enlarge","net":"catch","gun":"shoot",
        "scalpel":"incise","sensor":"detect","battery":"store_charge",
        # natural phenomena
        "sun":"shine","rain":"fall","wind":"blow","fire":"burn","water":"flow",
        "clock":"tick","river":"flow","star":"twinkle","moon":"reflect",
        "sun":"radiate","volcano":"erupt","earthquake":"shake",
        "glacier":"move","cloud":"float","snow":"cover","thunder":"boom",
        "lightning":"strike","tornado":"spin","hurricane":"destroy",
        # animal actions
        "dog":"bark","cat":"meow","horse":"neigh","cow":"moo","lion":"roar",
        "wolf":"howl","duck":"quack","chicken":"cluck","bird":"fly_bird",
        "fish":"swim","snake":"slither","rabbit":"hop","kangaroo":"jump",
        "eagle":"soar","owl":"hoot","parrot":"talk","frog":"croak",
        "cricket":"chirp","bear":"growl","elephant":"trumpet_call",
        "bee":"buzz","frog":"ribbit","donkey":"bray","goat":"bleat",
        "mouse":"squeak","pig":"oink","turkey":"gobble","crow":"caw",
        "hawk":"screech","dove":"coo","coyote":"yelp","bat":"echolocate",
        "whale":"sing","dolphin":"click","turtle":"retreat","chameleon":"camouflage",
        "spider":"spin_web","beaver":"build_dam","ant":"carry","termite":"eat_wood",
        "butterfly":"flutter","dragonfly":"hover","firefly":"glow",
        # transportation
        "car":"drive","bicycle":"ride","boat":"sail","rocket":"launch",
        "plane":"fly","train":"chug","bus":"transport","motorcycle":"speed",
        "helicopter":"hover","submarine":"dive","ship":"sail","truck":"haul",
        # technology
        "robot":"compute","sensor":"detect","battery":"charge","solar_panel":"generate",
        "printer":"print","scanner":"scan","router":"route_traffic",
        "server":"serve_data","database":"store_data","algorithm":"process",
        "solar_cell":"convert_light","motor":"rotate","engine":"combust",
        "turbine":"spin_generator","generator":"produce_electricity",
        # plants
        "flower":"bloom","tree":"grow","seed":"germinate","plant":"photosynthesize",
        "root":"absorb_water","leaf":"photosynthesize","fruit":"ripen",
        "cactus":"store_water","fern":"spore","moss":"cover_ground",
        "vine":"climb","weed":"spread","grass":"cover_soil",
    }
    for agent, action in function_agents.items():
        if agent in ALL_WORDS and action in ALL_WORDS:
            all_seqs.append({"sequence": [agent, action], "relations": [FUNCTION]})

    # 4. CAUSE pairs
    cause_pairs = [
        # natural → effect
        ("fire","heat"),("sun","light"),("rain","growth"),("wind","erosion"),
        ("cold","freeze"),("heat","melt"),("friction","heat"),
        ("storm","damage"),("cloud","rain"),("volcano","eruption"),
        ("drought","famine"),("flood","damage"),("sunburn","pain"),
        ("earthquake","destruction"),("rain","flood"),
        ("snow","freeze"),("ice","slipperiness"),("wind","wave"),
        ("sun","evaporation"),("cold","frost"),("heat","evaporation"),
        ("lightning","fire"),("tornado","destruction"),("hurricane","flood"),
        ("tsunami","destruction"),("avalanche","burial"),("mudslide","burial"),
        ("erosion","landslide"),("glacier","erosion"),
        # weather chain
        ("cloud","rain"),("rain","flood"),("wind","storm"),
        ("storm","lightning"),("cold","snow"),("heat","thunderstorm"),
        ("humidity","fog"),("pressure","wind"),
        # biological cause
        ("seed","sprout"),("sunlight","photosynthesis"),("water","plant_growth"),
        ("virus","illness"),("bacteria","infection"),("pollen","allergy"),
        ("exercise","sweat"),("exercise","strength"),("sleep","rest"),
        ("food","energy"),("water","hydration"),("medicine","healing"),
        ("sun","vitamin_d"),("stress","disease"),("aging","wrinkle"),
        ("smoke","cough"),("alcohol","intoxication"),("gravity","fall"),
        ("electricity","shock"),("pressure","compression"),
        ("poison","death"),("infection","fever"),("inflammation","pain"),
        ("bleeding","weakness"),("dehydration","thirst"),("hunger","weakness"),
        ("allergen","sneeze"),("pathogen","disease"),
        # psychological cause
        ("love","joy"),("hate","anger"),("loss","sadness"),
        ("success","happiness"),("failure","disappointment"),
        ("insult","anger"),("danger","fear"),("surprise","shock"),
        ("threat","anxiety"),("isolation","loneliness"),
        ("achievement","pride"),("kindness","gratitude"),
        ("music","emotion"),("art","inspiration"),
        ("meditation","calm"),("exercise","endorphin"),
        ("sunlight","happiness"),("darkness","fear"),
        # learning cause
        ("study","knowledge"),("practice","skill"),("training","endurance"),
        ("education","wisdom"),("reading","knowledge"),
        ("experiment","discovery"),("observation","understanding"),
        ("repetition","mastery"),("failure","learning"),
        # chain cause
        ("deforestation","erosion"),("fossil_fuel","pollution"),
        ("pollution","disease"),("overfishing","depletion"),
        ("urbanization","habitat_loss"),("climate_change","drought"),
        ("drought","crop_failure"),("crop_failure","famine"),
        ("poverty","crime"),("inequality","unrest"),
        ("war","destruction"),("conflict","suffering"),
        # physics cause
        ("gravity","weight"),("acceleration","force"),
        ("electric_current","magnetism"),("magnetism","attraction"),
        ("combustion","energy"),("nuclear_fission","energy"),
        ("solar_radiation","heat"),("friction","static_electricity"),
        ("sound_vibration","hearing"),("light_reflection","vision"),
        # chemistry/material cause
        ("oxidation","rust"),("acid","corrosion"),
        ("moisture","mold"),("heat","expansion"),
        ("cold","contraction"),("pressure","compression"),
        ("sunlight","bleaching"),("water","erosion"),
        # technology cause
        ("electricity","light"),("data","information"),
        ("algorithm","automation"),("automation","efficiency"),
        ("software","computation"),("network","connectivity"),
        ("hardware","performance"),("encryption","security"),
    ]
    for a,b in cause_pairs:
        if a in ALL_WORDS and b in ALL_WORDS:
            all_seqs.append({"sequence": [a,b], "relations": [CAUSE]})

    # 5. PART_OF pairs
    part_pairs = [
        # body hierarchy
        ("finger","hand"),("hand","arm"),("arm","body"),("toe","foot"),
        ("foot","leg"),("leg","body"),("ear","head"),("eye","head"),
        ("nose","head"),("mouth","head"),("head","body"),("heart","body"),
        ("lung","body"),("liver","body"),("kidney","body"),("brain","head"),
        ("skin","body"),("bone","body"),("muscle","body"),
        ("stomach","body"),("intestine","body"),
        ("thumb","hand"),("palm","hand"),("wrist","arm"),
        ("ankle","leg"),("knee","leg"),("hip","body"),
        ("shoulder","body"),("neck","body"),("spine","body"),
        ("rib","body"),("skull","head"),("jaw","head"),
        ("tongue","mouth"),("lip","mouth"),("tooth","mouth"),
        ("eyebrow","face"),("eyelid","eye"),("pupil","eye"),
        ("artery","circulatory_system"),("vein","circulatory_system"),
        ("capillary","circulatory_system"),("nerve","nervous_system"),
        ("neuron","brain"),("lobe","brain"),("cortex","brain"),
        # botanical parts
        ("leaf","tree"),("branch","tree"),("trunk","tree"),("root","tree"),
        ("petal","flower"),("stem","plant"),("seed","fruit"),
        ("thorn","rose"),("bark","tree"),("sap","tree"),
        ("pollen","flower"),("nectar","flower"),("bud","flower"),
        ("stalk","plant"),("pod","legume"),("kernel","grain"),
        ("pulp","fruit"),("peel","fruit"),("core","apple"),
        # mechanical parts
        ("wheel","car"),("door","car"),("engine","car"),("seat","car"),
        ("tire","car"),("steering_wheel","car"),("brake","car"),
        ("headlight","car"),("taillight","car"),("bumper","car"),
        ("hood","car"),("trunk","car"),("mirror","car"),
        ("gear","transmission"),("clutch","transmission"),
        ("propeller","plane"),("wing","plane"),("cockpit","plane"),
        ("blade","helicopter"),("rotor","helicopter"),
        # tech parts
        ("screen","phone"),("button","phone"),("key","keyboard"),
        ("sensor","robot"),("screen","laptop"),("trackpad","laptop"),
        ("chip","computer"),("cpu","computer"),("ram","computer"),
        ("hard_drive","computer"),("monitor","computer"),
        ("camera","phone"),("speaker","phone"),("microphone","phone"),
        ("battery","phone"),("charger","battery"),
        ("antenna","radio"),("lens","camera"),
        ("key","piano"),("pedal","piano"),("string","guitar"),
        ("button","remote"),("cord","phone"),
        ("blade","fan"),("blade","knife"),
        # book parts
        ("page","book"),("chapter","book"),("cover","book"),
        ("spine","book"),("index","book"),("glossary","book"),
        # building parts
        ("room","house"),("floor","building"),("wall","building"),
        ("roof","house"),("door","house"),("window","house"),
        ("stair","building"),("balcony","house"),("garage","house"),
        ("foundation","building"),("column","building"),
        ("pipe","plumbing"),("wire","electrical_system"),
        # animal parts
        ("wing","bird"),("beak","bird"),("feather","bird"),("tail","animal"),
        ("claw","animal"),("horn","animal"),("hoof","horse"),
        ("fin","fish"),("scale","fish"),("gill","fish"),
        ("whisker","cat"),("paw","dog"),("mane","lion"),
        ("trunk","elephant"),("tusk","elephant"),("shell","turtle"),
        ("antenna","insect"),("compound_eye","insect"),
        ("stinger","bee"),("web","spider"),
        # boat parts
        ("sail","boat"),("oar","boat"),("rudder","boat"),
        ("mast","boat"),("deck","boat"),("hull","boat"),
        ("anchor","boat"),("cabin","boat"),
        # tool parts
        ("handle","tool"),("head","hammer"),
        ("blade","saw"),("handle","screwdriver"),
        ("bit","drill"),("tip","knife"),
        # furniture parts
        ("leg","table"),("drawer","desk"),("shelf","bookcase"),
        ("armrest","chair"),("cushion","sofa"),("mattress","bed"),
        ("headboard","bed"),("frame","mirror"),
        # natural feature parts
        ("peak","mountain"),("slope","hill"),("bank","river"),
        ("mouth","river"),("source","river"),("tributary","river"),
        ("crater","volcano"),("lava","volcano"),
        ("shore","lake"),("bed","ocean"),("wave","ocean"),
        # food parts
        ("crust","pizza"),("topping","pizza"),
        ("yolk","egg"),("shell","egg"),
        ("slice","bread"),("crumb","bread"),
        ("meat","bone"),("fat","meat"),
    ]
    for a,b in part_pairs:
        if a in ALL_WORDS and b in ALL_WORDS:
            all_seqs.append({"sequence": [a,b], "relations": [PART_OF]})

    # 6. LOCATED_IN pairs
    loc_pairs = [
        # animals in habitat
        ("fish","water"),("bird","sky"),("whale","ocean"),("cactus","desert"),
        ("polar_bear","arctic"),("penguin","antarctic"),("camel","desert"),
        ("monkey","jungle"),("bear","forest"),("eagle","mountain"),
        ("lion","savanna"),("tiger","jungle"),("shark","ocean"),
        ("parrot","tropical_forest"),("koala","eucalyptus_forest"),
        ("deer","forest"),("wolf","forest"),("fox","forest"),
        ("rabbit","meadow"),("squirrel","tree"),("mouse","field"),
        ("bat","cave"),("owl","tree"),("hawk","sky"),
        ("dolphin","ocean"),("seal","coast"),("crab","beach"),
        ("frog","pond"),("turtle","pond"),("snake","grass"),
        ("lizard","desert"),("beaver","river"),("otter","river"),
        ("moose","lake"),("goat","mountain"),("ant","ground"),
        ("bee","flower"),("butterfly","garden"),("spider","web"),
        # people in places
        ("teacher","school"),("doctor","hospital"),("chef","kitchen"),
        ("doctor","clinic"),("teacher","classroom"),("chef","restaurant"),
        ("artist","studio"),("librarian","library"),
        ("worker","factory"),("soldier","barracks"),
        ("judge","court"),("lawyer","court"),("scientist","laboratory"),
        ("pilot","cockpit"),("driver","cab"),("sailor","ship"),
        ("astronaut","spaceship"),("farmer","field"),
        ("carpenter","workshop"),("plumber","basement"),
        ("student","school"),("professor","university"),
        ("musician","stage"),("singer","stage"),("actor","theater"),
        ("patient","hospital"),("athlete","gym"),
        ("cashier","store"),("banker","bank"),
        # vehicles in places
        ("car","road"),("plane","airport"),("train","station"),
        ("boat","harbor"),("ship","port"),("bicycle","bike_lane"),
        ("truck","highway"),("bus","bus_stop"),("helicopter","helipad"),
        ("submarine","ocean"),("rocket","launch_pad"),
        ("car","garage"),("plane","hangar"),
        # objects in places
        ("book","library"),("food","kitchen"),
        ("book","shelf"),("food","refrigerator"),
        ("money","bank"),("clothes","closet"),
        ("tools","workshop"),("plant","pot"),
        ("fish","aquarium"),("fruit","tree"),
        # natural features
        ("sun","sky"),("moon","sky"),("star","space"),
        ("iceberg","ocean"),("volcano","island"),
        ("mountain","land"),("river","valley"),
        ("lake","forest"),("waterfall","mountain"),
        ("cave","hill"),("forest","continent"),
        ("island","ocean"),("peninsula","coast"),
        ("desert","continent"),("plain","continent"),
        # artificial in places
        ("flower","garden"),("weed","field"),
        ("tree","park"),("bench","park"),
        ("statue","museum"),("painting","gallery"),
        ("computer","office"),("robot","factory"),
        ("sensor","robot_body"),("camera","ceiling"),
        ("lamp","room"),("mirror","bathroom"),
        ("bed","bedroom"),("sofa","living_room"),
        ("stove","kitchen"),("shower","bathroom"),
        ("desk","classroom"),("board","classroom"),
        ("telescope","observatory"),("microscope","laboratory"),
        # cultural
        ("pirate","ship"),("king","castle"),("queen","palace"),
        ("monk","monastery"),("nun","convent"),
        ("chef","restaurant"),("artist","studio"),
        # weather / phenomena
        ("rain","sky"),("snow","sky"),("cloud","sky"),
        ("fog","city"),("smoke","chimney"),
        ("rainbow","sky"),("aurora","sky"),
    ]
    for a,b in loc_pairs:
        if a in ALL_WORDS and b in ALL_WORDS:
            all_seqs.append({"sequence": [a,b], "relations": [LOCATED_IN]})

    # 7. OPPOSITE pairs
    opp_pairs = [
        # temperature
        ("hot","cold"),("warm","cool"),("boiling","freezing"),
        # size
        ("big","small"),("large","tiny"),("huge","miniature"),
        ("vast","limited"),("enormous","minute"),("broad","narrow"),
        # speed
        ("fast","slow"),("quick","sluggish"),("rapid","gradual"),
        ("hasty","leisurely"),("swift","ponderous"),
        # altitude / position
        ("high","low"),("above","below"),("top","bottom"),
        ("up","down"),("rise","fall"),("elevated","sunken"),
        ("above_ground","underground"),
        # light
        ("light","dark"),("bright","dim"),("shiny","dull"),
        ("illuminated","shadowy"),("radiant","murky"),
        # time
        ("day","night"),("morning","evening"),("dawn","dusk"),
        ("early","late"),("ancient","modern"),("past","future"),
        ("old","young"),("new","old"),
        # moisture
        ("wet","dry"),("damp","arid"),("soaked","parched"),
        ("humid","desert_dry"),("moist","dehydrated"),
        # cleanliness
        ("clean","dirty"),("pure","contaminated"),("sterile","infected"),
        ("tidy","messy"),("orderly","chaotic"),
        # wealth
        ("rich","poor"),("wealthy","destitute"),
        ("affluent","impoverished"),("prosperous","bankrupt"),
        # emotion
        ("happy","sad"),("joyful","miserable"),("cheerful","gloomy"),
        ("love","hate"),("friend","enemy"),("ally","adversary"),
        ("optimistic","pessimistic"),("hopeful","despairing"),
        # existence
        ("life","death"),("birth","death"),
        ("alive","dead"),("living","deceased"),
        # physical
        ("strong","weak"),("powerful","feeble"),
        ("robust","fragile"),("sturdy","flimsy"),
        ("hard","soft"),("solid","liquid"),("rigid","flexible"),
        ("tough","tender"),("firm","gentle"),
        ("thick","thin"),("fat","skinny"),("wide","narrow"),
        # state
        ("open","close"),("open","shut"),
        ("start","finish"),("begin","end"),("commence","conclude"),
        ("enter","exit"),("arrive","depart"),
        ("come","go"),("approach","retreat"),("advance","retreat"),
        # transactions
        ("buy","sell"),("give","take"),("lend","borrow"),
        ("offer","withdraw"),("supply","demand"),
        ("earn","spend"),("save","waste"),
        # creation
        ("create","destroy"),("build","demolish"),
        ("construct","dismantle"),("assemble","disassemble"),
        ("make","break"),("form","deform"),
        # capacity
        ("full","empty"),("filled","vacant"),("crowded","deserted"),
        ("packed","sparse"),("populated","uninhabited"),
        # weight
        ("heavy","light_weight"),("massive","weightless"),
        ("dense","airy"),("solid","hollow"),
        # edges
        ("sharp","dull"),("pointed","blunt"),("keen","blunt"),
        ("jagged","smooth"),
        # taste
        ("sweet","sour"),("bitter","sweet"),
        ("savory","bland"),("spicy","mild"),
        # sound
        ("loud","quiet"),("noisy","silent"),("deafening","faint"),
        ("boisterous","calm"),("raucous","peaceful"),
        # texture
        ("rough","smooth"),("coarse","fine"),("bumpy","flat"),
        ("uneven","level"),("rugged","sleek"),
        ("light_weight","heavy"),
        # dimension
        ("deep","shallow"),("profound","superficial"),
        ("long","short"),("tall","short"),("wide","narrow"),
        ("high","low"),
        # moral
        ("kind","cruel"),("good","evil"),("virtuous","wicked"),
        ("brave","cowardly"),("generous","stingy"),
        ("honest","deceitful"),("loyal","traitorous"),
        ("moral","immoral"),("ethical","unethical"),
        ("gentle","violent"),("patient","impatient"),
        # outcomes
        ("victory","defeat"),("success","failure"),
        ("win","lose"),("triumph","disaster"),
        ("profit","loss"),("gain","decline"),
        # health
        ("health","illness"),("healthy","sick"),
        ("strong","weak"),("fit","unfit"),
        ("well","unwell"),("healed","wounded"),
        # society
        ("peace","war"),("order","chaos"),
        ("civilization","anarchy"),("harmony","discord"),
        ("freedom","captivity"),("liberty","oppression"),
        ("independence","dependence"),("sovereignty","subjugation"),
        # movement
        ("ascend","descend"),("rise","fall"),
        ("climb","descend"),("fly","land"),
        ("accelerate","decelerate"),("speed","slow"),
        # trade
        ("import","export"),("buy","sell"),
        ("wholesale","retail"),
        # logic
        ("include","exclude"),("accept","reject"),
        ("admit","deny"),("allow","forbid"),
        ("permit","prohibit"),("enable","disable"),
        # opposites of mind
        ("awake","asleep"),("conscious","unconscious"),
        ("alert","drowsy"),("aware","oblivious"),
        # physics
        ("attract","repel"),("push","pull"),
        ("compress","expand"),("contract","expand"),
        ("inflate","deflate"),("charge","discharge"),
        ("connect","disconnect"),("attach","detach"),
        ("tie","untie"),("fasten","loosen"),
        ("lock","unlock"),("tune","detune"),
    ]
    for a,b in opp_pairs:
        if a in ALL_WORDS and b in ALL_WORDS:
            all_seqs.append({"sequence": [a,b], "relations": [OPPOSITE]})

    # 8. INSTRUMENT triplets (agent uses tool to do action)
    instrument_triplets = [
        # construction / woodwork
        ("carpenter","hammer","pound"),("carpenter","drill","make_hole"),
        ("carpenter","saw","cut"),("carpenter","plane","smooth"),
        ("carpenter","chisel","carve"),("carpenter","nail","join"),
        ("carpenter","level","measure"),("carpenter","tape","measure"),
        ("lumberjack","axe","chop"),("lumberjack","saw","cut"),
        # kitchen / food
        ("chef","knife","slice"),("chef","spatula","flip"),
        ("chef","whisk","beat"),("chef","rolling_pin","flatten"),
        ("chef","grater","shred"),("chef","peeler","peel"),
        ("chef","colander","drain"),("chef","blender","blend"),
        ("chef","stove","cook"),("chef","oven","bake"),
        # writing / drawing
        ("writer","pen","write"),("writer","pencil","write"),
        ("writer","keyboard","type"),("writer","typewriter","type"),
        ("artist","brush","paint"),("artist","pencil","sketch"),
        ("artist","charcoal","draw"),("artist","canvas","paint_on"),
        ("artist","easel","hold_canvas"),
        # farming
        ("farmer","shovel","dig"),("farmer","hoe","weed"),
        ("farmer","plow","till"),("farmer","tractor","plow_field"),
        ("farmer","scythe","harvest"),("farmer","rake","gather"),
        ("farmer","watering_can","water_plants"),
        ("gardener","rake","gather"),("gardener","pruner","trim"),
        ("gardener","shovel","dig"),("gardener","hose","water"),
        # mechanics
        ("mechanic","wrench","tighten"),("mechanic","screwdriver","turn"),
        ("mechanic","pliers","grip"),("mechanic","jack","lift"),
        ("mechanic","ratchet","loosen"),("mechanic","socket","tighten_bolt"),
        ("mechanic","welder","weld"),("mechanic","grease","lubricate"),
        # technology
        ("programmer","computer","code"),("programmer","keyboard","type"),
        ("engineer","cad","design"),("designer","mouse","click"),
        ("designer","tablet","draw"),("designer","stylus","sketch"),
        # medicine
        ("doctor","stethoscope","examine"),("doctor","scalpel","incise"),
        ("doctor","syringe","inject"),("doctor","thermometer","measure_temp"),
        ("surgeon","scalpel","operate"),("surgeon","forceps","grasp"),
        ("surgeon","suture","stitch_wound"),
        ("dentist","drill","drill_tooth"),("dentist","xray","examine_tooth"),
        ("veterinarian","stethoscope","examine_animal"),
        # music
        ("musician","guitar","play_music"),("musician","piano","play_piano"),
        ("musician","violin","play_violin"),("musician","drums","drum"),
        ("musician","flute","play_flute"),("musician","trumpet","play_trumpet"),
        ("musician","saxophone","play_saxophone"),("musician","cello","play_cello"),
        # science
        ("scientist","microscope","observe"),("scientist","beaker","mix"),
        ("scientist","pipette","measure_liquid"),("scientist","centrifuge","separate"),
        ("astronomer","telescope","observe"),("astronomer","spectrometer","analyze_light"),
        ("biologist","microscope","examine_cell"),("chemist","bunsen_burner","heat"),
        # photography / art
        ("photographer","camera","photo"),("photographer","lens","zoom"),
        ("photographer","tripod","stabilize"),("photographer","flash","illuminate"),
        ("sculptor","chisel","sculpt"),("sculptor","hammer","chip"),
        ("sculptor","mallet","carve"),("potter","wheel","throw_pottery"),
        ("potter","kiln","fire_pottery"),
        # teaching / office
        ("teacher","chalk","teach"),("teacher","whiteboard","write_on"),
        ("teacher","book","teach_from"),("teacher","projector","present"),
        ("student","book","learn"),("student","pencil","write"),
        ("student","calculator","compute"),("student","laptop","study"),
        # fishing / hunting
        ("fisherman","net","catch_fish"),("fisherman","rod","fish"),
        ("fisherman","hook","catch"),("fisherman","bait","attract_fish"),
        ("hunter","gun","shoot"),("hunter","bow","shoot_arrow"),
        ("hunter","trap","capture"),
        # soldier / police
        ("soldier","gun","shoot"),("soldier","rifle","aim"),
        ("soldier","binoculars","observe"),("soldier","radar","detect"),
        ("police","handcuffs","restrain"),("police","radio","communicate"),
        ("police","badge","identify"),
        # pilot / transportation
        ("pilot","controls","fly_plane"),("pilot","radar","navigate"),
        ("pilot","yoke","steer_plane"),("pilot","throttle","control_speed"),
        ("sailor","compass","navigate"),("sailor","wheel","steer_boat"),
        ("sailor","anchor","moor"),
        ("driver","steering_wheel","drive"),("driver","pedal","accelerate"),
        # investigation
        ("detective","magnifying_glass","investigate"),
        ("detective","flashlight","search"),("detective","notebook","record"),
        ("detective","camera","document"),
        # cleaning
        ("janitor","broom","sweep"),("janitor","mop","clean_floor"),
        ("janitor","vacuum","clean_carpet"),("janitor","rag","wipe"),
        # personal care
        ("barber","scissors","cut_hair"),("barber","comb","style_hair"),
        ("barber","razor","shave"),
        ("tailor","needle","sew"),("tailor","scissors","cut_fabric"),
        ("tailor","thread","stitch"),("tailor","measuring_tape","measure_body"),
        # sports
        ("athlete","ball","throw"),("athlete","racket","hit"),
        ("athlete","bat","swing"),("athlete","glove","catch"),
        ("golfer","club","hit_ball"),("tennis_player","racket","serve"),
        ("baseball_player","bat","hit_pitch"),
        ("boxer","gloves","punch"),("fencer","foil","thrust"),
    ]
    for a,t,act in instrument_triplets:
        if a in ALL_WORDS and t in ALL_WORDS and act in ALL_WORDS:
            all_seqs.append({"sequence": [a,t,act], "relations": [INSTRUMENT, NEXT]})

    # 9. Chains (NEXT): reduced — just enough for sequential context
    domain_lists = [ANIMALS, BIRDS, FISH, REPTILES, INSECTS, PLANTS, VEGETABLES,
                   FRUITS, FOODS, BEVERAGES, TOOLS, PROFESSIONS, BODY_PARTS,
                   EMOTIONS, SPORTS, GEOGRAPHY, WEATHER, TECH, VEHICLES,
                   COLORS, SHAPES, ABSTRACT, MUSIC, CLOTHING, FURNITURE]
    for _ in range(50):
        dl = random.choice(domain_lists)
        if len(dl) < 3:
            continue
        chain = random.sample(dl, min(random.randint(2,5), len(dl)))
        all_seqs.append({"sequence": chain, "relations": [NEXT] * (len(chain)-1)})

    # 10. HAS_PROPERTY
    property_pairs = [
        # color
        ("rose","red"),("sky","blue"),("grass","green"),("snow","white"),
        ("gold","yellow"),("orange_fruit","orange_color"),("banana","yellow"),
        ("apple","red"),("grape","purple"),("lemon","yellow"),
        ("coal","black"),("chalk","white"),("brick","red"),
        ("sun","yellow"),("moon","silver"),("ocean","blue"),
        ("forest","green"),("desert","brown"),("blood","red"),
        ("leaf","green"),("fire","orange"),("night","black"),
        # temperature
        ("ice","cold"),("fire","hot"),("sun","hot"),
        ("snow","cold"),("lava","hot"),("desert","hot"),
        ("arctic","cold"),("coffee","hot"),("steam","hot"),
        # taste / flavor
        ("lemon","sour"),("sugar","sweet"),("coffee","bitter"),
        ("honey","sweet"),("salt","salty"),("pepper","spicy"),
        ("ginger","spicy"),("chocolate","sweet"),("vinegar","sour"),
        ("grapefruit","bitter"),("mint","fresh"),
        # size
        ("elephant","big"),("mouse","small"),("whale","huge"),
        ("ant","tiny"),("mountain","tall"),("tree","tall"),
        ("giraffe","tall"),("dwarf","short"),
        ("planet","vast"),("atom","tiny"),("ocean","vast"),
        # speed
        ("cheetah","fast"),("turtle","slow"),("snail","slow"),
        ("rabbit","fast"),("rocket","fast"),("horse","fast"),
        ("greyhound","fast"),("sloth","slow"),
        # hardness / texture
        ("diamond","hard"),("pillow","soft"),("steel","hard"),
        ("iron","strong"),("glass","fragile"),("rubber","elastic"),
        ("rock","hard"),("butter","soft"),("wood","hard"),
        ("cloth","soft"),("metal","hard"),("feather","soft"),
        ("concrete","hard"),("clay","soft"),("sponge","porous"),
        # weight
        ("lead","heavy"),("feather","light_weight"),
        ("iron","heavy"),("cotton","light_weight"),
        ("gold","heavy"),("helium","light_weight"),
        ("stone","heavy"),("paper","light_weight"),
        ("steel","heavy"),("air","light_weight"),
        ("water","heavy"),("foam","light_weight"),
        # pattern
        ("tiger","striped"),("zebra","striped"),("leopard","spotted"),
        ("giraffe","spotted"),("ladybug","spotted"),
        ("zebra","striped"),("clownfish","striped"),
        ("dalmation","spotted"),("butterfly","colorful"),
        ("peacock","colorful"),("parrot","colorful"),
        # domestication / nature
        ("wolf","wild"),("dog","domestic"),("cat","domestic"),
        ("lion","wild"),("cow","domestic"),("horse","domestic"),
        ("tiger","wild"),("sheep","domestic"),("fox","wild"),
        ("bear","wild"),("chicken","domestic"),("deer","wild"),
        # surface
        ("silk","smooth"),("sandpaper","rough"),("ice","slippery"),
        ("glass","smooth"),("bark","rough"),("velvet","soft"),
        ("concrete","rough"),("marble","smooth"),
        # sharpness
        ("knife","sharp"),("needle","sharp"),("razor","sharp"),
        ("sword","sharp"),("scissors","sharp"),
        # shape / form
        ("cactus","spiky"),("rose","thorny"),
        ("ball","round"),("planet","round"),("sun","round"),
        ("box","square"),("table","flat"),("balloon","round"),
        # consistency
        ("water","liquid"),("ice","solid"),("steam","gaseous"),
        ("water","wet"),("sand","dry"),("rock","dry"),
        ("honey","viscous"),("water","fluid"),
        ("clay","malleable"),("stone","hard"),
        # brightness
        ("night","dark"),("sun","bright"),
        ("star","bright"),("moon","bright"),
        ("cave","dark"),("tunnel","dark"),
        ("diamond","shiny"),("mirror","reflective"),
        # age
        ("turtle","old"),("mountain","ancient"),
        ("baby","young"),("puppy","young"),
        ("ruin","ancient"),("technology","modern"),
        # strength
        ("diamond","strong"),("steel","strong"),("iron","strong"),
        ("thread","weak"),("paper","weak"),("glass","fragile"),
        ("chain","strong"),("rope","strong"),
        # intelligence / emotion
        ("dolphin","intelligent"),("human","intelligent"),
        ("dog","loyal"),("cat","independent"),
        ("lion","brave"),("mouse","timid"),
        ("fox","cunning"),("donkey","stubborn"),
        # utility
        ("knife","useful"),("phone","useful"),
        ("internet","useful"),("education","valuable"),
    ]
    for a,b in property_pairs:
        if a in ALL_WORDS and b in ALL_WORDS:
            all_seqs.append({"sequence": [a,b], "relations": [HAS_PROPERTY]})

    # 11. IS_A for every word → every ancestor category as individual pair
    def get_all_ancestors(word):
        cat = CATEGORY_MAP.get(word)
        ancestors = []
        while cat:
            ancestors.append(cat)
            cat = SUPER_CATEGORIES.get(cat)
        return ancestors

    for word in sorted(ALL_WORDS):
        ancestors = get_all_ancestors(word)
        for anc in ancestors:
            if anc in ALL_WORDS and word != anc:
                all_seqs.append({"sequence": [word, anc], "relations": [IS_A]})
        # Also add word → its top-level group labels
        for group_name, members in [
            ("animal", ANIMALS + BIRDS + FISH + REPTILES + INSECTS),
            ("plant", PLANTS + VEGETABLES + FRUITS),
            ("artifact", TOOLS + TECH + VEHICLES + MUSIC + CLOTHING + FURNITURE),
            ("food", FOODS), ("beverage", BEVERAGES),
            ("body_part", BODY_PARTS), ("emotion", EMOTIONS),
            ("sport", SPORTS), ("color", COLORS), ("shape", SHAPES),
            ("professional", PROFESSIONS),
            ("geography", GEOGRAPHY), ("weather_phenomenon", WEATHER),
        ]:
            if word in members and group_name in ALL_WORDS:
                all_seqs.append({"sequence": [word, group_name], "relations": [IS_A]})

    # 13. Random walks — minimal (just enough for NEXT context)
    all_words_list = sorted(ALL_WORDS)
    for _ in range(100):
        length = random.randint(2, 4)
        chain = random.sample(all_words_list, min(length, len(all_words_list)))
        all_seqs.append({"sequence": chain, "relations": [NEXT] * (len(chain) - 1)})

    # ── Deduplicate ──
    seen = set()
    unique = []
    for s in all_seqs:
        key = json.dumps(s, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(s)

    random.shuffle(unique)

    vocab = set()
    for s in unique:
        for w in s["sequence"]:
            vocab.add(w)

    output = {
        "description": f"Dataset v8 EN large: {len(unique)} sequences, {len(vocab)} words, balanced relations",
        "sequences": unique,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Secuencias: {len(unique)}")
    print(f"Vocabulario: {len(vocab)} palabras")
    print(f"Guardado en: {output_path}")

    # Relation type distribution
    rel_counts = {}
    for s in unique:
        for r in (s.get("relations") or []):
            rel_counts[r] = rel_counts.get(r, 0) + 1
    print("Distribución de relaciones:")
    for r, c in sorted(rel_counts.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else r"D:\ma_brain_data\dataset_v8_en_large.json"
    main(path)
