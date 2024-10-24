from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

class Chatbot:
    def __init__(self):
        self.knowledge_base = {
            "preguntas": [

                # Saludo y cortesía (10)
                "hola", "¿cómo estás?", "¿cuál es tu nombre?", "adiós", "gracias", 
                "buenos días", "buenas noches", "¿qué tal?", "un placer", "hasta luego",
                
                # Tecnología (10)
                "¿qué es una computadora?", "¿qué es un smartphone?", "¿qué es inteligencia artificial?", 
                "¿qué es el aprendizaje automático?", "¿qué es un algoritmo?", "¿qué es la realidad virtual?", 
                "¿qué es un robot?", "¿qué es el internet de las cosas?", "¿qué es la ciberseguridad?", 
                "¿cómo funciona un servidor?",

                # Programación (10)
                "¿qué es la programación?", "¿qué es un lenguaje de programación?", 
                "¿qué es Python?", "¿qué es Java?", "¿qué es un bucle en programación?", 
                "¿qué es una función?", "¿qué es una variable?", "¿qué es un objeto?", 
                "¿qué es un framework?", "¿qué es una API?",

                # Deportes (10)
                "¿qué es el fútbol?", "¿qué es el baloncesto?", "¿qué es el tenis?", 
                "¿qué es el béisbol?", "¿quién es Messi?", "¿quién es LeBron James?", 
                "¿qué es una copa del mundo?", "¿qué es la NBA?", "¿qué es el Tour de Francia?", 
                "¿qué es un maratón?",

                # Cultura general (10)
                "¿quién fue Albert Einstein?", "¿qué es la gravedad?", "¿qué es el sistema solar?", 
                "¿qué es una galaxia?", "¿quién fue Leonardo da Vinci?", "¿qué es la historia?", 
                "¿qué es la filosofía?", "¿qué es la literatura?", "¿qué es el arte?", "¿qué es la música?",

                # Viajes (10)
                "¿qué es un destino turístico?", "¿qué es un pasaporte?", "¿cuáles son los mejores destinos turísticos?", 
                "¿qué es el turismo sostenible?", "¿qué es una visa?", "¿cuál es la capital de Francia?", 
                "¿cuál es la capital de Japón?", "¿qué es un hotel?", "¿qué es una aerolínea?", "¿cómo reservar un vuelo?",

                # Comida (10)
                "¿qué es la comida rápida?", "¿qué es una pizza?", "¿qué es la gastronomía?", 
                "¿qué es una dieta balanceada?", "¿qué es un chef?", "¿qué es una receta?", 
                "¿qué es el sushi?", "¿qué es la cocina mexicana?", "¿qué es la cocina italiana?", "¿qué es el vino?",

                # Ciencia (10)
                "¿qué es la física?", "¿qué es la química?", "¿qué es la biología?", 
                "¿qué es la matemática?", "¿qué es la medicina?", "¿qué es un átomo?", 
                "¿qué es la evolución?", "¿qué es el ADN?", "¿qué es la energía?", "¿qué es la electricidad?",

                # Música (10)
                "¿qué es el rock?", "¿qué es el jazz?", "¿qué es la música clásica?", 
                "¿quién es Beethoven?", "¿quién es Mozart?", "¿quién es The Beatles?", 
                "¿qué es un instrumento musical?", "¿qué es una orquesta?", "¿qué es una banda musical?", 
                "¿qué es un concierto?",

                # Cine (10)
                "¿qué es una película?", "¿quién es Steven Spielberg?", "¿qué es Hollywood?", 
                "¿qué es una serie de televisión?", "¿qué es un actor?", "¿qué es un director de cine?", 
                "¿qué es un guion?", "¿qué es el Oscar?", "¿quién es Leonardo DiCaprio?", "¿qué es un tráiler de película?"
            ],
            "respuestas": [
      
                 # Saludo y cortesía (10)
                "¡Hola! ¿En qué puedo ayudarte?", "Estoy bien, gracias por preguntar. ¿Y tú?", "Soy un IA Hibrida de ayuda. Puedes llamarme BlockCortex.", 
                "¡Hasta luego! Que tengas un buen día.", "¡De nada! ¿Hay algo más en lo que pueda ayudarte?", 
                "¡Buenos días! ¿En qué puedo ayudarte?", "Buenas noches. ¿Hay algo en lo que pueda asistirte?", 
                "Todo bien, gracias. ¿Y tú?", "¡El placer es mío!", "¡Nos vemos!",

                # Tecnología (10)
                "Una computadora es una máquina que procesa datos.", "Un smartphone es un teléfono inteligente.", 
                "La inteligencia artificial es la simulación de la inteligencia humana en máquinas.", 
                "El aprendizaje automático es un método de análisis de datos que automatiza la creación de modelos analíticos.", 
                "Un algoritmo es una secuencia de pasos para resolver un problema.", 
                "La realidad virtual es una experiencia simulada creada por tecnología.", "Un robot es una máquina que puede realizar tareas automáticamente.", 
                "El internet de las cosas es la interconexión de dispositivos a través de internet.", 
                "La ciberseguridad es la protección de sistemas informáticos.", "Un servidor es un sistema que proporciona recursos a otros sistemas.",

                # Programación (10)
                "La programación es la creación de instrucciones para que una computadora las ejecute.", "Un lenguaje de programación es una forma de comunicarse con las computadoras.", 
                "Python es un lenguaje de programación fácil de aprender y muy usado.", "Java es un lenguaje de programación orientado a objetos.", 
                "Un bucle es una estructura que repite un bloque de código varias veces.", "Una función es un bloque de código reutilizable.", 
                "Una variable es un espacio para almacenar información.", "Un objeto es una instancia de una clase en programación orientada a objetos.", 
                "Un framework es un conjunto de herramientas y bibliotecas que facilitan el desarrollo de software.", "Una API es una interfaz que permite la comunicación entre diferentes aplicaciones.",

                # Deportes (10)
                "El fútbol es un deporte jugado entre dos equipos de 11 jugadores con un balón.", "El baloncesto es un deporte jugado entre dos equipos de 5 jugadores que intentan encestar el balón.", 
                "El tenis es un deporte en el que dos jugadores (o parejas) usan raquetas para golpear una pelota.", "El béisbol es un deporte en el que dos equipos alternan entre bateo y defensa.", 
                "Messi es un famoso jugador de fútbol argentino.", "LeBron James es un famoso jugador de baloncesto estadounidense.", 
                "La Copa del Mundo es un torneo internacional de fútbol organizado por la FIFA.", "La NBA es la liga profesional de baloncesto en los Estados Unidos.", 
                "El Tour de Francia es una carrera anual de ciclismo en Francia.", "Un maratón es una carrera de larga distancia de 42,195 km.",

                # Cultura general (10)
                "Albert Einstein fue un físico famoso por su teoría de la relatividad.", "La gravedad es la fuerza que atrae los objetos con masa hacia el centro de la Tierra.", 
                "El sistema solar es el conjunto de planetas que orbitan alrededor del Sol.", "Una galaxia es un conjunto de estrellas, planetas y otros cuerpos celestes.", 
                "Leonardo da Vinci fue un artista e inventor del Renacimiento.", "La historia es el estudio de eventos pasados.", 
                "La filosofía es el estudio de las cuestiones fundamentales sobre la existencia, el conocimiento y la ética.", "La literatura es el arte de la palabra escrita.", 
                "El arte es una expresión creativa humana.", "La música es el arte de combinar sonidos de manera armónica.",

                # Viajes (10)
                "Un destino turístico es un lugar que atrae a los viajeros.", "Un pasaporte es un documento que permite viajar a diferentes países.", 
                "Algunos de los mejores destinos turísticos son París, Tokio y Nueva York.", "El turismo sostenible busca minimizar el impacto ambiental.", 
                "Una visa es un documento que permite la entrada a un país específico.", "La capital de Francia es París.", 
                "La capital de Japón es Tokio.", "Un hotel es un establecimiento que ofrece alojamiento temporal.", 
                "Una aerolínea es una empresa que ofrece transporte aéreo.", "Para reservar un vuelo, puedes usar una agencia de viajes o una plataforma en línea.",

                        # Respuestas de Comida
                "La comida rápida es aquella que se prepara y sirve rápidamente, como las hamburguesas.",
                "La pizza es un platillo italiano hecho con masa, salsa de tomate, queso y varios ingredientes.",
                "La gastronomía es el arte y ciencia de preparar alimentos y bebidas.",
                "Una dieta balanceada es aquella que incluye alimentos de todos los grupos en las proporciones correctas.",
                "Un chef es un profesional de la cocina que se encarga de preparar y supervisar la creación de alimentos.",
                "Una receta es una lista de ingredientes y pasos para preparar un platillo.",
                "El sushi es un platillo japonés que se prepara con arroz sazonado y pescado crudo u otros ingredientes.",
                "La cocina mexicana es famosa por su uso de maíz, chile y frijoles en una amplia variedad de platillos.",
                "La cocina italiana es conocida por su pasta, pizza y uso de ingredientes frescos como el tomate y el aceite de oliva.",
                "El vino es una bebida alcohólica fermentada hecha a partir de uvas.",

                # Respuestas de Ciencia
                "La física es la ciencia que estudia las propiedades de la materia y la energía.",
                "La química es la ciencia que estudia la composición y las propiedades de las sustancias.",
                "La biología es la ciencia que estudia los seres vivos.",
                "La matemática es la ciencia que estudia los números, formas y patrones.",
                "La medicina es la ciencia dedicada a la salud y el tratamiento de enfermedades.",
                "Un átomo es la unidad más pequeña de un elemento químico.",
                "La evolución es el proceso por el cual las especies cambian con el tiempo.",
                "El ADN es la molécula que contiene la información genética de los seres vivos.",
                "La energía es la capacidad de realizar trabajo o producir movimiento.",
                "La electricidad es una forma de energía producida por el movimiento de electrones.",

                # Respuestas de Música
                "El rock es un género musical que se caracteriza por el uso de guitarras eléctricas y una estructura rítmica fuerte.",
                "El jazz es un género musical nacido en Estados Unidos que se caracteriza por la improvisación.",
                "La música clásica es un estilo de música formal, que abarca compositores como Beethoven y Mozart.",
                "Beethoven fue un compositor alemán del periodo clásico y romántico, conocido por sus sinfonías.",
                "Mozart fue un compositor austriaco del periodo clásico, considerado uno de los más grandes músicos de todos los tiempos.",
                "The Beatles fue una banda de rock británica que revolucionó la música en los años 60.",
                "Un instrumento musical es cualquier objeto que produce sonido cuando se toca, como el piano o la guitarra.",
                "Una orquesta es un conjunto de músicos que tocan instrumentos, principalmente de cuerdas, viento y percusión.",
                "Una banda musical es un grupo de músicos que tocan juntos, comúnmente con instrumentos como guitarras y batería.",
                "Un concierto es una presentación en vivo de música ante una audiencia.",

                # Respuestas de Cine
                "Una película es una obra audiovisual que narra una historia a través de imágenes y sonido.",
                "Steven Spielberg es un director de cine estadounidense conocido por películas como 'Jurassic Park' y 'E.T.'.",
                "Hollywood es una industria cinematográfica ubicada en Los Ángeles, conocida por ser el epicentro del cine mundial.",
                "Una serie de televisión es una producción audiovisual dividida en episodios que cuentan una historia.",
                "Un actor es una persona que interpreta un papel en una película o serie.",
                "Un director de cine es la persona encargada de supervisar la creación artística de una película.",
                "Un guion es el texto escrito que contiene los diálogos y descripciones de una película.",
                "El Oscar es un premio cinematográfico otorgado por la Academia de las Artes y Ciencias Cinematográficas.",
                "Leonardo DiCaprio es un actor de cine estadounidense conocido por películas como 'Titanic' y 'El Renacido'.",
                "Un tráiler de película es un breve adelanto que muestra escenas de una película próxima a estrenarse."

            ]
        }
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(self.knowledge_base["preguntas"])
    
    def get_response(self, user_input):
        user_vector = self.vectorizer.transform([user_input.lower()])
        similarities = cosine_similarity(user_vector, self.X)
        most_similar = np.argmax(similarities)
        if similarities[0][most_similar] < 0.3:
            return "Lo siento, no entiendo esa pregunta. ¿Podrías reformularla?"
        return self.knowledge_base["respuestas"][most_similar]
