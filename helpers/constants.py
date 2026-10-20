class Features:
    ID = "id"
    AGE = "Age"
    FLIGHT_DISTANCE = "Flight Distance"
    INFLIGHT_WIFI_SERVICE = "Inflight wifi service"
    DEPARTURE_ARRIVAL_TIME_CONVENIENT = "Departure/Arrival time convenient"
    EASE_OF_ONLINE_BOOKING = "Ease of Online booking"
    GATE_LOCATION = "Gate location"
    FOOD_AND_DRINK = "Food and drink"
    ONLINE_BOARDING = "Online boarding"
    SEAT_COMFORT = "Seat comfort"
    INFLIGHT_ENTERTAINMENT = "Inflight entertainment"
    ON_BOARD_SERVICE = "On-board service"
    LEG_ROOM_SERVICE = "Leg room service"
    BAGGAGE_HANDLING = "Baggage handling"
    CHECKIN_SERVICE = "Checkin service"
    CLEANLINESS = "Cleanliness"
    DEPARTURE_DELAY_IN_MINUTES = "Departure Delay in Minutes"
    ARRIVAL_DELAY_IN_MINUTES = "Arrival Delay in Minutes"
    GENDER = "Gender"
    CUSTOMER_TYPE = "Customer Type"
    TYPE_OF_TRAVEL = "Type of Travel"
    CLASS = "Class"
    

CATEGORICAL_FEATURES = [
    Features.GENDER, #binary
    Features.CUSTOMER_TYPE, #binary
    Features.TYPE_OF_TRAVEL, #binary
    Features.CLASS, #multiclass
]
    

TARGET = "satisfaction"
