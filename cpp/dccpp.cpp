// DigitalCurlingPy Cpp Module

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <CurlingSimulator.h>

pybind11::array_t<float>
simulate(const pybind11::array_t<float>& stone,
         const int turn,
         const float vx,
         const float vy,
         const int spin)
{
    GAMESTATE state;
    SHOTVEC shot;
    
    state.ShotNum = turn;
    state.WhiteToMove = turn % 2;
    const float *ploc = stone.data();
    for (int i = 0; i < turn; ++i)
    {
        state.body[i][0] = *ploc; ploc++;
        state.body[i][1] = *ploc; ploc++;
    }
    shot.x = vx;
    shot.y = vy;
    shot.angle = spin;
    int ret = Simulation(&state, shot);
    
    pybind11::array_t<float> next_stone({16, 2});
    float *pnloc = next_stone.mutable_data();
    for (int i = 1; i <= turn; ++i)
    {
        *pnloc = state.body[i][0]; pnloc++;
        *pnloc = state.body[i][1]; pnloc++;
    }
    return next_stone;
}

PYBIND11_MODULE(dccpp, m) {
    m.doc() = "cpp module for DigitalCurlingPy library"; // optional module docstring
    m.def("simulate", &simulate, "A function which moves stones");
}