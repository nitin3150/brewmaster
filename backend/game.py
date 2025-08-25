class CreateGameSession:
    def __init__(self): 
        self.turn = 0
        self.competetion_mode = False
        self.game_state = {
            'turn': 0,
            'competition_mode': False,
            # Human Team (Green)
            'green_team_profit': 100000,
            'green_team_inventory': 100,
            'green_team_price': 10,
            'green_team_production': 50,
            'green_team_marketing': 500,
            'green_team_projected_demand': 50,
            'green_team_profit_this_turn': 0,
            # Mesa Team (Blue)
            'blue_team_profit': 100000,
            'blue_team_inventory': 100,
            'blue_team_price': 10,
            'blue_team_production': 50,
            'blue_team_marketing': 500,
            'blue_team_projected_demand': 50,
            'blue_team_profit_this_turn': 0,
            # Temporal Team (Purple)
            'purple_team_profit': 100000,
            'purple_team_inventory': 100,
            'purple_team_price': 10,
            'purple_team_production': 50,
            'purple_team_marketing': 500,
            'purple_team_projected_demand': 50,
            'purple_team_profit_this_turn': 0,
            # Google ADK Team (Orange)
            'orange_team_profit': 100000,
            'orange_team_inventory': 100,
            'orange_team_price': 10,
            'orange_team_production': 50,
            'orange_team_marketing': 500,
            'orange_team_projected_demand': 50,
            'orange_team_profit_this_turn': 0,
            # Event log
            'event_log': ["Game Started!", "Competition: Human vs Mesa vs Temporal vs Google ADK"],
            # History
            'sales_history': {'green': [], 'blue': [], 'purple': [], 'orange': []},
            'production_history': {'green': [], 'blue': [], 'purple': [], 'orange': []},
            'price_history': {'green': [], 'blue': [], 'purple': [], 'orange': []}
        }