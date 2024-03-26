import numpy as np
import matplotlib.pyplot as plt

# macierz przejścia
t1 = ['P', 'K', 'N']

# macierz wystąpień
wyst = np.array([[2, 4, 0], [0, 0, 4], [4, 0, 2]])

# strategia przeciwnika
prob_op = np.array([0.5, 0.1, 0.4])

n = 500
enemy_last_choice = 'P'

payout = 0
payout_history = [payout]

for i in range(n):
    player_choice = np.random.choice(t1, p=wyst[t1.index(enemy_last_choice)] / sum(wyst[t1.index(enemy_last_choice)]))
    print(player_choice)

    enemy_choice = np.random.choice(t1, p=prob_op)

    if enemy_choice == 'P':
        wyst[t1.index(enemy_last_choice)][t1.index('N')] += 1
    elif enemy_choice == 'K':
        wyst[t1.index(enemy_last_choice)][t1.index('P')] += 1
    else:  # enemy choice == 'N'
        wyst[t1.index(enemy_last_choice)][t1.index('K')] += 1

    if enemy_choice == player_choice:
        print("Draw")

    elif (player_choice == 'P' and enemy_choice == 'K') or (
            player_choice == 'K' and enemy_choice == 'N') or (
            player_choice == 'N' and enemy_choice == 'P'):
        print("Player wins!")
        payout += 1
    else:
        print("Enemy wins!")
        payout -= 1
    payout_history.append(payout)
    enemy_last_choice = enemy_choice

plt.figure(num="Payout over time", figsize=(10, 6))
plt.plot(payout_history)
plt.axhline(0, color='g', linestyle='--')
plt.xlabel('n')
plt.title("Payout over n rounds")
plt.ylabel('Payout')
plt.grid(True)
plt.show()
