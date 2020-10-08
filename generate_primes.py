from math import sqrt


number_to_create = 100_000


def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    i = 3
    while i <= sqrt(n):
        if n % i == 0:
            return False
        i = i + 2

    return True


def prime_generator():
    n = 1
    while True:
        n += 1
        if is_prime(n):
            yield n


generator = prime_generator()

with open('positives.txt', 'w') as writer:
    for i in range(number_to_create):
        writer.write(f'{next(generator)}\n')
    
primes = list()    
for i in range(number_to_create):
    primes.append(next(generator))

non_primes = list()
for i in range(primes[-1]):
    non_primes.append(i)

print('length of non_primes before removal', len(non_primes))

for i in primes:
    if i in non_primes:
        non_primes.remove(i)

print('length of non_primes after removal ', len(non_primes))

with open('negatives.txt', 'w') as writer:
    for i in non_primes:
        writer.write(f'{i}\n')
