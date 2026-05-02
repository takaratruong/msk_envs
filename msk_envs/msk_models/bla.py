
import numpy as np
def main():
    vec1 = (0.077230736607309336,0.029190024656100086,0.0074348929980000032)
    vec2 = (0.023005830644321993,-0.015584531118749998,-0.0052443478100000001)

    vec_add = (0.063970169355678008,-0.011406010125810932,0.0084340478099999942)

    vec1, vec2 = np.array(vec1), np.array(vec2)
    vec_add = np.array(vec_add)
    print(vec1 + vec2 + vec_add)

if __name__ == "__main__":
    main()