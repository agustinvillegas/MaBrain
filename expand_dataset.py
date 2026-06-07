import json
import itertools

# Relations
IS_A = "IS_A"
NEXT = "NEXT"
PROP = "PROPERTY"
HAS = "HAS"
IS_ATRIBUTO = "IS_ATRIBUTO"
HAS_CARACTERISTICA = "HAS_CARACTERISTICA"
NEXT_OF = "NEXT_OF"
PROPERTY_OF = "PROPERTY_OF"
HAS_INV = "HAS_INVERSE"
ATRIBUTO_DE = "ATRIBUTO_DE"
CARACTERISTICA_DE = "CARACTERISTICA_DE"
CAUSA = "CAUSA"
FUNCION = "FUNCION"
PARTE_DE = "PARTE_DE"
INSTRUMENTO = "INSTRUMENTO"
OPUESTO = "OPUESTO"
SIMILAR_A = "SIMILAR_A"
UBICADO_EN = "UBICADO_EN"
EFECTO_DE = "EFECTO_DE"
FUNCION_DE = "FUNCION_DE"
TIENE_PARTE = "TIENE_PARTE"
CONTIENE = "CONTIENE"

INVERSE_MAP = {
    IS_A: HAS,
    NEXT: NEXT_OF,
    PROP: PROPERTY_OF,
    HAS: HAS_INV,
    IS_ATRIBUTO: ATRIBUTO_DE,
    HAS_CARACTERISTICA: CARACTERISTICA_DE,
    CAUSA: EFECTO_DE,
    FUNCION: FUNCION_DE,
    PARTE_DE: TIENE_PARTE,
    UBICADO_EN: CONTIENE,
}

def seq(words, relations=None):
    return {"sequence": words, "relations": relations} if relations else {"sequence": words}

def chain(head, *tail, rel=NEXT):
    return seq([head] + list(tail), [rel] * len(tail))

def isa(member, *categories):
    return seq([member] + list(categories), [IS_A] * len(categories))

def cause(cause_word, effect_word):
    return seq([cause_word, effect_word], [CAUSA])

def funcion(organ, action):
    return seq([organ, action], [FUNCION])

def parte_de(part, whole):
    return seq([part, whole], [PARTE_DE])

def instrumento(agent, tool, action):
    return seq([agent, tool, action], [INSTRUMENTO, NEXT])

def opuesto(a, b):
    return seq([a, b], [OPUESTO])

def similar(a, b):
    return seq([a, b], [SIMILAR_A])

def ubicado_en(entity, location):
    return seq([entity, location], [UBICADO_EN])

# ================= ANIMALS =================
SUBGROUPS = {
    "perro": ["canido", "carnivoro"],
    "lobo": ["canido", "carnivoro"],
    "zorro": ["canido", "carnivoro"],
    "gato": ["felino", "carnivoro"],
    "leon": ["felino", "carnivoro"],
    "tigre": ["felino", "carnivoro"],
    "caballo": ["equino", "herbivoro"],
    "vaca": ["bovino", "herbivoro"],
    "oveja": ["ovino", "herbivoro"],
    "cerdo": ["suido", "omnivoro"],
}

GROUP_ANCESTORS = {
    "mamifero": ["vertebrado", "animal", "ser_vivo"],
    "ave": ["vertebrado", "animal", "ser_vivo"],
    "pez": ["vertebrado", "animal", "ser_vivo"],
    "reptil": ["vertebrado", "animal", "ser_vivo"],
    "insecto": ["invertebrado", "animal", "ser_vivo"],
}

PROPERTY_CHARACTERISTICS = {
    "domestico": "compania",
    "salvaje": "independencia",
    "herbivoro": "vegetal",
    "carnivoro": "carne",
    "volador": "cielo",
    "no_volador": "suelo",
    "migratorio": "viaje",
    "acuatico": "agua",
}

ANIMALS = {
    "mamiferos": {
        "members": {
            "perro":    {"sounds": "ladra",  "actions": "corre", "body": "pata"},
            "gato":     {"sounds": "maulla", "actions": "salta", "body": "garra"},
            "leon":     {"sounds": "ruge",   "actions": "caza",  "body": "melena"},
            "tigre":    {"sounds": "gruñe",  "actions": "acecha","body": "raya"},
            "elefante": {"sounds": "barrita","actions": "carga", "body": "trompa"},
            "caballo":  {"sounds": "relincha","actions":"galopa", "body": "crin"},
            "vaca":     {"sounds": "muge",   "actions": "pasta", "body": "ubre"},
            "oveja":    {"sounds": "bala",   "actions": "pastar","body": "lana"},
            "cerdo":    {"sounds": "gruñe",  "actions": "hoza",  "body": "hocico"},
            "mono":     {"sounds": "chilla", "actions": "trepa", "body": "cola"},
            "oso":      {"sounds": "gruñe",  "actions": "pesca", "body": "pelaje"},
            "delfin":   {"sounds": "chirria","actions": "salta", "body": "aleta"},
            "ballena":  {"sounds": "canta",  "actions": "bucea", "body": "aleta"},
            "jirafa":   {"sounds": "bufa",   "actions": "alcanza","body":"cuello"},
            "cebra":    {"sounds": "relincha","actions":"corre",  "body": "raya"},
            "conejo":   {"sounds": "chilla", "actions": "salta", "body": "oreja"},
            "ardilla":  {"sounds": "chirria","actions": "salta", "body": "cola"},
            "zorro":    {"sounds": "aulla",  "actions": "acecha","body": "cola"},
            "lobo":     {"sounds": "aulla",  "actions": "caza",  "body": "colmillos"},
            "hamster":  {"sounds": "chilla", "actions": "corre", "body": "bolsa"},
        },
        "base": "mamifero",
        "properties": ["domestico", "salvaje", "herbivoro", "carnivoro"],
    },
    "aves": {
        "members": {
            "aguila":   {"sounds": "chilla", "actions": "vuela", "body": "ala"},
            "pajaro":   {"sounds": "canta",  "actions": "vuela", "body": "pluma"},
            "bubo":     {"sounds": "ulula",  "actions": "caza",  "body": "pluma"},
            "pato":     {"sounds": "parpa",  "actions": "nada",  "body": "pico"},
            "gallo":    {"sounds": "cacarea","actions": "canta", "body": "cresta"},
            "gallina":  {"sounds": "cloquea","actions": "picotea","body": "pico"},
            "loro":     {"sounds": "habla",  "actions": "vuela", "body": "pluma"},
            "pinguino": {"sounds": "grazna", "actions": "nada",  "body": "ala"},
            "cisne":    {"sounds": "silba",  "actions": "nada",  "body": "pluma"},
            "pavo":     {"sounds": "gluglutea","actions":"desfila","body":"pluma"},
        },
        "base": "ave",
        "properties": ["volador", "no_volador", "migratorio", "acuatico"],
    },
    "peces": {
        "members": {
            "tiburon":  {"sounds": None,     "actions": "caza",  "body": "aleta"},
            "salmón":   {"sounds": None,     "actions": "rema",  "body": "escama"},
            "dorado":   {"sounds": None,     "actions": "nada",  "body": "escama"},
            "merluza":  {"sounds": None,     "actions": "nada",  "body": "escama"},
            "atun":     {"sounds": None,     "actions": "nada",  "body": "aleta"},
        },
        "base": "pez",
        "properties": ["marino", "rio", "depredador"],
    },
    "reptiles": {
        "members": {
            "serpiente":{"sounds": "silva",  "actions": "repta", "body": "escama"},
            "cocodrilo":{"sounds": "bufa",   "actions": "acecha","body": "escama"},
            "lagarto":  {"sounds": "silva",  "actions": "repta", "body": "escama"},
            "tortuga":  {"sounds": "bufa",   "actions": "nada",  "body": "caparazon"},
            "iguana":   {"sounds": "silva",  "actions": "trepa", "body": "escama"},
        },
        "base": "reptil",
        "properties": ["venenoso", "no_venenoso", "terrestre", "acuatico"],
    },
    "insectos": {
        "members": {
            "hormiga":  {"sounds": None,     "actions": "trabaja","body":"antena"},
            "abeja":    {"sounds": "zumba",  "actions": "poliniza","body":"ala"},
            "mariposa": {"sounds": None,     "actions": "vuela", "body": "ala"},
            "mosca":    {"sounds": "zumba",  "actions": "vuela", "body": "ala"},
            "araña":    {"sounds": None,     "actions": "teje",  "body": "tela"},
        },
        "base": "insecto",
        "properties": ["volador", "no_volador", "social"],
    },
}

def generate_animal_sequences():
    s = []
    for group_name, group in ANIMALS.items():
        base = group["base"]
        ancestors = GROUP_ANCESTORS.get(base, ["animal", "ser_vivo"])
        for name, traits in group["members"].items():
            subgroups = SUBGROUPS.get(name, [])
            full_chain = [name] + subgroups + [base] + ancestors
            s.append(isa(*full_chain))
            if traits.get("sounds"):
                s.append(chain(name, traits["sounds"], rel=NEXT))
            if traits.get("actions"):
                s.append(chain(name, traits["actions"], rel=NEXT))
            if traits.get("body"):
                s.append(chain(name, traits["body"], rel=HAS))
        for prop in group["properties"]:
            for name in group["members"]:
                s.append(chain(name, prop, rel=HAS))
            s.append(isa(prop, "atributo"))
            if prop in PROPERTY_CHARACTERISTICS:
                s.append(chain(prop, PROPERTY_CHARACTERISTICS[prop], rel=HAS_CARACTERISTICA))
    # cross connections
    for g1, g2 in itertools.combinations(ANIMALS.values(), 2):
        for m1 in list(g1["members"])[:3]:
            for m2 in list(g2["members"])[:3]:
                s.append(chain("animal", m1, m2, rel=NEXT))
                s.append(chain(m1, m2, rel=NEXT))
    return s

# ================ PLANTAS =================
PLANT_GROUPS = {
    "arboles":   {"base":"arbol", "ancestors":["planta","ser_vivo"], "members":["roble","pino","sauce","cerezo","manzano"]},
    "flores":    {"base":"flor", "ancestors":["planta","ser_vivo"], "members":["rosa","girasol","margarita","orquidea","tulipan"]},
    "frutas":    {"base":"fruta", "ancestors":["vegetal","planta","ser_vivo"], "members":["manzana","naranja","platano","uva","fresa"]},
    "verduras":  {"base":"verdura", "ancestors":["vegetal","planta","ser_vivo"], "members":["zanahoria","brocoli","tomate","lechuga","cebolla"]},
}

def generate_plant_sequences():
    s = []
    for cat, group in PLANT_GROUPS.items():
        base = group["base"]
        ancestors = group["ancestors"]
        for member in group["members"]:
            full_chain = [member] + [base] + ancestors
            s.append(isa(*full_chain))
    return s

# ================ CUERPO =================
CUERPO = {
    "cabeza":   ["ojo", "oreja", "nariz", "boca", "lengua", "diente", "cerebro",
                 "pelo", "frente", "mejilla", "barbilla", "ceja", "pestana"],
    "torso":    ["corazon", "pulmon", "higado", "estomago", "riñon",
                 "hueso", "piel", "musculo", "sangre"],
    "extremidades": ["mano", "brazo", "codo", "hombro", "dedo", "pierna",
                     "rodilla", "pie", "talón"],
}

def generate_body_sequences():
    s = []
    for group, parts in CUERPO.items():
        for part in parts:
            group_word = group.replace("s", "")
            s.append(isa(part, group_word, "parte_del_cuerpo", "cuerpo", "humano"))
            s.append(parte_de(part, group_word))
    # Funciones de órganos y partes
    s.append(funcion("ojo", "ver"))
    s.append(funcion("oreja", "escuchar"))
    s.append(funcion("nariz", "oler"))
    s.append(funcion("lengua", "gustar"))
    s.append(funcion("piel", "tocar"))
    s.append(funcion("corazon", "latir"))
    s.append(funcion("pulmon", "respirar"))
    s.append(funcion("cerebro", "pensar"))
    s.append(funcion("mano", "agarrar"))
    s.append(funcion("pie", "caminar"))
    s.append(funcion("boca", "hablar"))
    s.append(funcion("diente", "masticar"))
    s.append(funcion("estomago", "digerir"))
    s.append(funcion("higado", "filtrar"))
    s.append(funcion("riñon", "filtrar"))
    s.append(funcion("musculo", "contraer"))
    s.append(funcion("sangre", "oxigenar"))
    # PARTE_DE para los grupos principales
    s.append(parte_de("cabeza", "cuerpo"))
    s.append(parte_de("torso", "cuerpo"))
    s.append(parte_de("extremidades", "cuerpo"))
    return s

# ================ NATURALEZA =================
NATURALEZA = {
    "elementos": {"agua": ["liquido", "fluye", "moja", "vida"],
                  "fuego": ["calor", "quema", "llama", "luz"],
                  "tierra": ["solida", "suelo", "cultivo", "vida"],
                  "aire": ["invisible", "viento", "respira", "vida"]},
    "clima":    {"lluvia": ["agua", "nube", "moja", "tormenta"],
                 "nieve": ["frio", "blanco", "hielo", "invierno"],
                 "viento": ["aire", "fuerte", "tormenta", "empuja"]},
    "geografia": {"rio":   ["agua", "fluye", "largo", "pesca"],
                  "mar":   ["agua", "salado", "olas", "profundo"],
                  "selva": ["arbol", "verde", "animal", "humedo"],
                  "desierto": ["arena", "seco", "calor", "cactus"]},
}

def generate_nature_sequences():
    s = []
    for cat, items in NATURALEZA.items():
        for name, props in items.items():
            for p in props:
                s.append(chain(name, p, rel=NEXT))
            s.append(isa(name, cat.replace("s",""), "naturaleza", "mundo"))
    # Relaciones CAUSA
    s.append(cause("lluvia", "rio_crece"))
    s.append(cause("lluvia", "inundacion"))
    s.append(cause("fuego", "calor"))
    s.append(cause("fuego", "ceniza"))
    s.append(cause("viento", "erosion"))
    s.append(cause("sol", "luz"))
    s.append(cause("sol", "calor"))
    s.append(cause("nube", "lluvia"))
    s.append(cause("nieve", "frio"))
    s.append(cause("nieve", "hielo"))
    s.append(cause("rio", "fertilidad"))
    s.append(cause("mar", "sal"))
    s.append(cause("viento", "ola"))
    s.append(cause("sequia", "desierto"))
    s.append(cause("fotosintesis", "oxigeno"))
    return s

# ================ TECNOLOGIA =================
TECH_GROUPS = {
    "dispositivos": {"base":"dispositivo_digital", "ancestors":["herramienta","tecnologia"], "members":["computadora","telefono","tablet","televisor","reloj_inteligente"]},
    "internet": {"base":"internet", "ancestors":["red","tecnologia"], "members":["navegador","correo","red_social","busqueda","descarga"]},
    "software": {"base":"software", "ancestors":["programa","tecnologia"], "members":["programa","aplicacion","sistema_operativo","base_datos","algoritmo"]},
    "partes_computadora": {"base":"componente", "ancestors":["hardware","tecnologia"], "members":["procesador","memoria","disco_duro","pantalla","teclado","mouse"]},
}

def generate_tech_sequences():
    s = []
    for cat, group in TECH_GROUPS.items():
        base = group["base"]
        ancestors = group["ancestors"]
        for member in group["members"]:
            full_chain = [member] + [base] + ancestors
            s.append(isa(*full_chain))
    # FUNCION
    s.append(funcion("computadora", "procesar"))
    s.append(funcion("computadora", "calcular"))
    s.append(funcion("telefono", "comunicar"))
    s.append(funcion("navegador", "explorar"))
    s.append(funcion("programa", "ejecutar"))
    s.append(funcion("procesador", "calculo"))
    s.append(funcion("memoria", "almacenar"))
    s.append(funcion("disco_duro", "guardar"))
    s.append(funcion("pantalla", "mostrar"))
    s.append(funcion("teclado", "escribir"))
    s.append(funcion("mouse", "navegar"))
    s.append(funcion("aplicacion", "automatizar"))
    s.append(funcion("algoritmo", "resolver"))
    s.append(funcion("base_datos", "organizar"))
    s.append(funcion("sistema_operativo", "gestionar"))
    # INSTRUMENTO
    s.append(instrumento("usuario", "teclado", "escribir"))
    s.append(instrumento("usuario", "mouse", "navegar"))
    s.append(instrumento("usuario", "pantalla", "ver"))
    s.append(instrumento("programador", "computadora", "programar"))
    s.append(instrumento("disenador", "aplicacion", "crear"))
    # NEXT (algunos extras)
    s.append(chain("computadora", "internet", rel=NEXT))
    s.append(chain("telefono", "internet", rel=NEXT))
    return s

# ================ EMOCIONES =================
EMOCIONES = {
    "positivas":    {"alegria": "sonreir",   "amor": "abrazar", "esperanza": "soñar",
                     "gratitud": "agradecer", "paz": "meditar"},
    "negativas":    {"tristeza": "llorar",    "miedo": "huir",   "ira": "gritar",
                     "ansiedad": "preocupar", "soledad": "aislar"},
}

def generate_emotion_sequences():
    s = []
    for cat, items in EMOCIONES.items():
        for emotion, action in items.items():
            s.append(isa(emotion, f"emocion_{cat}", "emocion", "humano"))
            s.append(chain(emotion, action, rel=NEXT))
    return s

# ================ ALIMENTOS =================
ALIMENTOS = {
    "bebidas":     ["agua", "jugo", "cafe", "te", "leche"],
    "comidas":     ["pan", "arroz", "pasta", "carne", "queso", "huevo"],
    "cocina":      ["cocinar", "hervir", "freir", "hornear", "picar"],
    "sabores":     ["dulce", "salado", "amargo", "acido", "umami"],
}

def generate_food_sequences():
    s = []
    for cat, items in ALIMENTOS.items():
        for item in items:
            s.append(isa(item, cat.replace("s",""), "alimento", "necesidad"))
    s.append(chain("cocinar", "comida", rel=NEXT))
    s.append(chain("hervir", "agua", rel=NEXT))
    s.append(chain("freir", "aceite", rel=NEXT))
    s.append(chain("pan", "comer", rel=NEXT))
    s.append(chain("cafe", "beber", rel=NEXT))
    s.append(chain("dulce", "azucar", rel=NEXT))
    s.append(chain("salado", "sal", rel=NEXT))
    return s

# ================ DEPORTES =================
DEPORTES = {
    "pelota":    ["futbol", "tenis", "basquetbol", "volibol", "beisbol"],
    "agua":      ["natacion", "buceo", "surf", "remo"],
    "aire":      ["vuelo", "paracaidismo", "ala_delta"],
    "combate":   ["boxeo", "karate", "judo", "esgrima"],
}

def generate_sport_sequences():
    s = []
    for cat, sports in DEPORTES.items():
        for sport in sports:
            s.append(isa(sport, f"deporte_{cat}", "deporte", "actividad", "humano"))
    s.append(chain("futbol", "patear", rel=NEXT))
    s.append(chain("tenis", "golpear", rel=NEXT))
    s.append(chain("natacion", "nadar", rel=NEXT))
    s.append(chain("boxeo", "golpear", rel=NEXT))
    return s

# ================ ESPACIO =================
ESPACIO = {
    "astros":     {"sol": "estrella", "luna": "satelite", "tierra": "planeta",
                   "marte": "planeta", "jupiter": "planeta"},
    "fenomenos":  ["gravedad", "luz", "calor", "radiacion", "magnetismo"],
}

def generate_space_sequences():
    s = []
    for name, kind in ESPACIO["astros"].items():
        s.append(isa(name, kind, "astro", "universo"))
    s.append(isa("estrella", "astro", "universo"))
    s.append(isa("planeta", "astro", "universo"))
    s.append(isa("satelite", "astro", "universo"))
    for phenom in ESPACIO["fenomenos"]:
        s.append(isa(phenom, "fenomeno_fisico", "universo"))
    s.append(chain("sol", "luz", rel=NEXT))
    s.append(chain("sol", "calor", rel=NEXT))
    s.append(chain("luna", "reflejar", rel=NEXT))
    s.append(chain("tierra", "orbitar", rel=NEXT))
    s.append(chain("gravedad", "atraer", rel=NEXT))
    return s

# ================ SOCIEDAD =================
SOCIEDAD = {
    "profesiones": {
        "base":"profesion",
        "ancestors":["ocupacion","humano"],
        "members": ["medico","maestro","ingeniero","abogado","artista",
                    "bombero","policia","cocinero","piloto","escritor"],
        "actions": {"medico":"curar","maestro":"enseñar","ingeniero":"construir",
                    "abogado":"defender","artista":"crear","bombero":"apagar",
                    "policia":"proteger","cocinero":"cocinar","piloto":"volar",
                    "escritor":"escribir"},
        "intermediate": {
            "medico": ["profesional_salud"],
            "maestro": ["educador"],
            "ingeniero": ["profesional_tecnico"],
            "abogado": ["profesional_legal"],
            "artista": ["creativo"],
            "bombero": ["servicio_emergencia"],
            "policia": ["servicio_emergencia"],
            "cocinero": ["gastronomo"],
            "piloto": ["aviador"],
            "escritor": ["literato"],
        }
    },
    "lugares": {
        "base":"lugar",
        "ancestors":["sociedad"],
        "members": ["escuela","hospital","biblioteca","parque","museo",
                    "teatro","restaurante","banco","tienda","oficina"],
    },
}

def generate_society_sequences():
    s = []
    prof_group = SOCIEDAD["profesiones"]
    base = prof_group["base"]
    ancestors = prof_group["ancestors"]
    for member in prof_group["members"]:
        intermediate = prof_group.get("intermediate",{}).get(member, [])
        full_chain = [member] + intermediate + [base] + ancestors
        s.append(isa(*full_chain))
    # FUNCION (antes NEXT)
    for prof, action in prof_group["actions"].items():
        s.append(funcion(prof, action))
    # UBICADO_EN
    s.append(ubicado_en("medico", "hospital"))
    s.append(ubicado_en("maestro", "escuela"))
    s.append(ubicado_en("ingeniero", "oficina"))
    s.append(ubicado_en("abogado", "oficina"))
    s.append(ubicado_en("bombero", "estacion_bomberos"))
    s.append(ubicado_en("policia", "comisaria"))
    s.append(ubicado_en("cocinero", "restaurante"))
    s.append(ubicado_en("piloto", "aeropuerto"))
    s.append(ubicado_en("escritor", "biblioteca"))
    s.append(ubicado_en("artista", "museo"))
    # INSTRUMENTO
    s.append(instrumento("escritor", "computadora", "escribir"))
    s.append(instrumento("ingeniero", "computadora", "disenar"))
    s.append(instrumento("medico", "instrumento_quirurgico", "operar"))
    s.append(instrumento("artista", "pincel", "pintar"))
    s.append(instrumento("cocinero", "cuchillo", "cortar"))
    # LUGARES (ISA + FUNCION + PARTE_DE)
    place_group = SOCIEDAD["lugares"]
    for place in place_group["members"]:
        s.append(isa(place, place_group["base"], *place_group["ancestors"]))
    s.append(funcion("escuela", "educar"))
    s.append(funcion("hospital", "sanar"))
    s.append(funcion("biblioteca", "prestar"))
    s.append(funcion("parque", "recrear"))
    s.append(funcion("museo", "exponer"))
    s.append(funcion("restaurante", "alimentar"))
    s.append(funcion("banco", "guardar_dinero"))
    s.append(funcion("oficina", "trabajar"))
    s.append(funcion("teatro", "actuar"))
    s.append(funcion("tienda", "vender"))
    s.append(parte_de("aula", "escuela"))
    s.append(parte_de("sala_operaciones", "hospital"))
    s.append(parte_de("sala_lectura", "biblioteca"))
    return s

# ================ COLORES =================
COLORES = {
    "colores":   ["rojo", "azul", "verde", "amarillo", "negro", "blanco",
                  "naranja", "violeta", "rosa", "marron", "gris", "dorado"],
    "asociaciones": {"rojo": "pasion", "azul": "calma", "verde": "naturaleza",
                     "amarillo": "alegria", "negro": "oscuridad", "blanco": "paz"},
}

def generate_color_sequences():
    s = []
    for color in COLORES["colores"]:
        s.append(isa(color, "color", "propiedad", "concepto"))
    for color, meaning in COLORES["asociaciones"].items():
        s.append(chain(color, meaning, rel=HAS))
    return s

# ================ MATEMATICAS =================
MATEMATICAS = {
    "operaciones": {"sumar": "mas", "restar": "menos", "multiplicar": "por", "dividir": "entre"},
    "formas":      ["circulo", "cuadrado", "triangulo", "rectangulo", "rombo", "oval"],
    "numeros":     ["uno", "dos", "tres", "cuatro", "cinco", "diez", "cien"],
}

def generate_math_sequences():
    s = []
    for op, symbol in MATEMATICAS["operaciones"].items():
        s.append(isa(op, "operacion", "matematica", "ciencia"))
        s.append(chain(op, symbol, rel=NEXT))
    s.append(chain("sumar", "total", rel=NEXT))
    s.append(chain("restar", "diferencia", rel=NEXT))
    s.append(chain("multiplicar", "producto", rel=NEXT))
    s.append(chain("dividir", "cociente", rel=NEXT))
    for shape in MATEMATICAS["formas"]:
        s.append(isa(shape, "forma", "geometria", "matematica"))
    return s

# ================ ACCIONES FISICAS =================
ACCIONES_FISICAS = {
    "verbos": ["correr", "saltar", "nadar", "volar", "caminar", "rodar",
               "empujar", "tirar", "levantar", "cargar"],
    "resultados": {"correr": "velocidad", "saltar": "altura", "nadar": "agua",
                   "volar": "aire", "caminar": "movimiento", "rodar": "giro"},
}

def generate_action_sequences():
    s = []
    for verb in ACCIONES_FISICAS["verbos"]:
        s.append(isa(verb, "accion_fisica", "movimiento", "concepto"))
    for verb, result in ACCIONES_FISICAS["resultados"].items():
        s.append(chain(verb, result, rel=NEXT))
    # CAUSA
    s.append(cause("empujar", "mover"))
    s.append(cause("tirar", "mover"))
    s.append(cause("levantar", "elevar"))
    s.append(cause("cargar", "transportar"))
    s.append(cause("golpear", "romper"))
    s.append(cause("empujar", "caer"))
    # INSTRUMENTO
    s.append(instrumento("persona", "mano", "agarrar"))
    s.append(instrumento("persona", "pie", "caminar"))
    s.append(instrumento("persona", "cuchillo", "cortar"))
    s.append(instrumento("persona", "martillo", "golpear"))
    s.append(instrumento("persona", "llave", "abrir"))
    s.append(instrumento("persona", "lapiz", "escribir"))
    return s

# ================ ABSTRACTOS =================
ABSTRACTOS = {
    "conceptos": ["tiempo", "espacio", "energia", "materia", "informacion",
                  "verdad", "belleza", "justicia", "libertad", "conocimiento"],
    "pares": {"tiempo": "reloj", "espacio": "distancia", "energia": "fuerza",
              "materia": "masa", "informacion": "dato",
              "verdad": "hecho", "belleza": "arte", "justicia": "ley",
              "libertad": "derecho", "conocimiento": "sabiduria"},
}

def generate_abstract_sequences():
    s = []
    for concept in ABSTRACTOS["conceptos"]:
        s.append(isa(concept, "concepto_abstracto", "idea", "mente"))
    for concept, related in ABSTRACTOS["pares"].items():
        s.append(chain(concept, related, rel=HAS))
    return s

# ================ FISICA / CIENCIA =================
FISICA = {
    "fuerzas": {
        "gravedad": ["atraer", "caer", "masa"],
        "magnetismo": ["atraer", "metal", "iman"],
        "electricidad": ["cargar", "bombilla", "circuito"],
        "friccion": ["calor", "resistencia", "desgaste"],
    },
    "procesos": {
        "fotosintesis": ["luz", "clorofila", "glucosa"],
        "combustion": ["fuego", "calor", "dioxido"],
        "evaporacion": ["agua", "vapor", "calor"],
        "condensacion": ["vapor", "agua", "frio"],
        "fusion": ["solido", "liquido", "calor"],
        "solidificacion": ["liquido", "solido", "frio"],
    },
    "materiales": {
        "agua": ["liquido", "transparente", "vida", "h2o"],
        "hierro": ["metal", "duro", "conductor", "pesado"],
        "oro": ["metal", "brillante", "valioso", "blando"],
        "plastico": ["sintetico", "moldeable", "aislante", "ligero"],
    },
}

def generate_physics_sequences():
    s = []
    for cat, items in FISICA.items():
        for name, props in items.items():
            for p in props:
                s.append(chain(name, p, rel=NEXT))
            s.append(isa(name, cat.replace("s",""), "fenomeno_fisico", "ciencia", "conocimiento"))
    # CAUSA
    s.append(cause("gravedad", "caer"))
    s.append(cause("gravedad", "peso"))
    s.append(cause("magnetismo", "atraccion"))
    s.append(cause("electricidad", "luz"))
    s.append(cause("friccion", "calor"))
    s.append(cause("fotosintesis", "oxigeno"))
    s.append(cause("combustion", "calor"))
    s.append(cause("evaporacion", "vapor"))
    s.append(cause("condensacion", "lluvia"))
    s.append(cause("fusion", "liquido"))
    s.append(cause("solidificacion", "hielo"))
    s.append(cause("calor", "evaporacion"))
    s.append(cause("frio", "solidificacion"))
    # PARTE_DE (materiales -> propiedades)
    s.append(parte_de("electricidad", "fisica"))
    s.append(parte_de("magnetismo", "fisica"))
    s.append(parte_de("gravedad", "fisica"))
    s.append(parte_de("fotosintesis", "biologia"))
    s.append(parte_de("combustion", "quimica"))
    # FUNCION (herramientas cientificas)
    s.append(funcion("iman", "atraer"))
    s.append(funcion("bombilla", "iluminar"))
    s.append(funcion("circuito", "conducir"))
    return s

# ============================================================
# INVERSE GENERATION
# ============================================================
def generate_inverses(sequences, inverse_map):
    inverses = []
    seen = set()
    for entry in sequences:
        tokens = entry.get("sequence", [])
        rels = entry.get("relations")
        if rels is None or len(tokens) < 2:
            continue
        for i in range(len(tokens)-1):
            a, b = tokens[i], tokens[i+1]
            rel = rels[i]
            inv_rel = inverse_map.get(rel)
            if inv_rel is None:
                continue
            inv_pair = (b, a, inv_rel)
            if inv_pair not in seen:
                seen.add(inv_pair)
                inverses.append({"sequence": [b, a], "relations": [inv_rel]})
    return sequences + inverses

# ============================================================
# MAIN
# ============================================================
def main():
    all_sequences = []
    generators = [
        ("animales", generate_animal_sequences),
        ("plantas", generate_plant_sequences),
        ("cuerpo", generate_body_sequences),
        ("naturaleza", generate_nature_sequences),
        ("tecnologia", generate_tech_sequences),
        ("emociones", generate_emotion_sequences),
        ("alimentos", generate_food_sequences),
        ("deportes", generate_sport_sequences),
        ("espacio", generate_space_sequences),
        ("sociedad", generate_society_sequences),
        ("colores", generate_color_sequences),
        ("matematicas", generate_math_sequences),
        ("acciones", generate_action_sequences),
        ("abstractos", generate_abstract_sequences),
        ("fisica", generate_physics_sequences),
    ]
    for name, gen in generators:
        seqs = gen()
        all_sequences.extend(seqs)
    # Dedup
    seen = set()
    deduped = []
    for s in all_sequences:
        key = tuple(s["sequence"])
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    # Add inverse relations
    final = generate_inverses(deduped, INVERSE_MAP)
    # Stats
    vocab = set()
    for s in final:
        vocab.update(s["sequence"])
    dataset = {
        "name": "conceptos_rutas_v5",
        "description": f"Dataset v6 con {len(final)} secuencias, {len(vocab)} palabras, relaciones funcionales/causales/espaciales",
        "sequences": final,
    }
    with open("datasets/dataset_v6.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"  Secuencias: {len(final)}")
    print(f"  Palabras unicas: {len(vocab)}")
    print(f"  Dominios: {len(generators)}")
    print(f"  Guardado: datasets/dataset_v6.json")

if __name__ == "__main__":
    main()
