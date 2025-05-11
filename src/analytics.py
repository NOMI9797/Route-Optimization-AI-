def calculate_analytics(total_distance, avg_speed=60, fuel_efficiency=15, fuel_price=1.0):
    """
    Calculate route analytics metrics.
    Args:
        total_distance (float): Total distance in kilometers
        avg_speed (float): Average speed in km/h
        fuel_efficiency (float): Vehicle fuel efficiency in km/l
        fuel_price (float): Fuel price in $/l
    Returns:
        dict: Analytics metrics
    """
    estimated_time = total_distance / avg_speed  # in hours
    fuel_consumption = total_distance / fuel_efficiency  # in liters
    estimated_cost = fuel_consumption * fuel_price  # in $
    return {
        'total_distance': total_distance,
        'estimated_time': estimated_time,
        'fuel_consumption': fuel_consumption,
        'estimated_cost': estimated_cost
    } 