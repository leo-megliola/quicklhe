// lhe_parser.cpp
// Build: c++ -O2 -std=c++17 -shared -fPIC \
//   $(python3 -m pybind11 --includes) \
//   -lexpat \
//   -o lhe_parser$(python3-config --extension-suffix) \
//   lhe_parser.cpp

/*
Structure of event: 

<event>
[header] NUP IDPRUP XWGTUP SCALUP AQEDUP AQCDUP
[particle] IDUP ISTUP MOTHUP(1) MOTHUP(2) ICOLUP(1) ICOLUP(2) PUP(1) PUP(2) PUP(3) PUP(4) PUP(5) VTIMUP SPINUP
[particle] ...
.
.
.
</event>

Structure of init (gives some vital information about processes, Xsection, etc.):

<init>
[header] IDBMUP(1) IDBMUP(2) EBMUP(1) EBMUP(2) PDFGUP(1) PDFGUP(2) PDFSUP(1) PDFSUP(2) IDWTUP NPRUP
[process] XSECUP XERRUP XMAXUP LPRUP
[process] ...
.
.
.
</init>

i_evt : shape {n_events,   2}            cols: [NUP, IDPRUP]
f_evt : shape {n_events,   4+n_weights}  cols: [XWGTUP, SCALUP, AQEDUP, AQCDUP, wgt_0, wgt_1, ...]
i_ptc : shape {n_particles, 7}           cols: [evt_idx, IDUP, ISTUP, MOTHUP1, MOTHUP2, ICOLUP1, ICOLUP2]
f_ptc : shape {n_particles, 7}           cols: [PUP1, PUP2, PUP3, PUP4, PUP5, VTIMUP, SPINUP]
*/
#include <sstream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <cstring>
#include <charconv>
#include <string_view>
#include <expat.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Pass 1 – line scan to count events and weights
// ---------------------------------------------------------------------------
static std::tuple<int,int,int> countDimensions(const std::string& filename)
{
    std::ifstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + filename);

    int n_events    = 0;
    int n_weights   = 0;
    int n_particles = 0;
    int n_line = 0;

    std::string line;
    while (std::getline(f, line)) {
        ++n_line;
        if (line.find("<event>") != std::string::npos) {
            ++n_events;
            std::getline(f, line); ++n_line;
            std::istringstream iss(line);
            int n;
            if (!(iss >> n)) throw std::runtime_error("Failed to parse particle count from event header on line: " + std::to_string(n_line));
            n_particles += n;
        }
            
        if (line.find("<weight ") != std::string::npos) ++n_weights;
    }

    return {n_events, n_weights, n_particles};
}

// ---------------------------------------------------------------------------
// Pass 2 – SAX parse with expat to fill the array
// ---------------------------------------------------------------------------
static constexpr int NO_CAPTURE   = 0;
static constexpr int EVENT_HEADER = 1;
static constexpr int WGT_TAG      = 2;

struct ParseState
{
    double*     fevt_arr     = nullptr; 
    int*        ievt_arr     = nullptr;
    double*     fptc_arr     = nullptr;
    int*        iptc_arr     = nullptr;

    int         n_weights    = 0;        
    int         n_events     = 0;   
    int         n_particles  = 0;

    int         cur_event    = 0;        // current row index
    int         cur_weight   = 0;        // current column index (within event)
    int         cur_particle = 0;
    int         capture      = NO_CAPTURE;

    std::string charBuf;               // accumulates character data
};

// helper to traverse charBuf and extract int/double values with no copy or string convert
template <typename T>
bool consume_next(std::string_view& sv, T& value) {
    // advance past leading whitespace
    size_t begin = sv.find_first_not_of(" \t\n\r");
    if (begin == std::string_view::npos) return false; // exit if no non delim token
    sv.remove_prefix(begin);

    // find end of token
    size_t end = sv.find_first_of(" \t\n\r");
    std::string_view token = sv.substr(0, end);

    // use from_chars to convert to int/double
    auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
    
    // update string_view to point after this token
    if (end == std::string_view::npos) sv = {};
    else sv.remove_prefix(end);

    return ec == std::errc();
}

//process header and particles from event and put them directly into struct
void processEvent(ParseState* s)
{   
    std::string_view sv(s->charBuf);
 
    // read headder (careful to save n_ptc for looping condation below)
    int n_ptc = 0; 
    if (!consume_next(sv, n_ptc)) throw std::runtime_error("Failed to parse particle count from event number: " + std::to_string(s->cur_event));

    int* ie = s->ievt_arr + s->cur_event * 2;
    ie[0] = n_ptc;
    consume_next(sv, ie[1]);

    double* fe = s->fevt_arr + s->cur_event * (4 + s->n_weights);
    for (int i = 0; i < 4; i++) { // read doubles
        consume_next(sv, fe[i]);
    }

    // read particles 
    for (int p = 0; p < n_ptc; p++) {
        int*    ip = s->iptc_arr + s->cur_particle * 7;
        double* fp = s->fptc_arr + s->cur_particle * 7;
        ip[0] = s->cur_event;
        // first 6 ints
        for (int i = 1; i <= 6; i++) {
            consume_next(sv, ip[i]);
        }
        // remaining 7 doubles
        for (int i = 0; i < 7; ++i) {
            consume_next(sv, fp[i]);
        }
        s->cur_particle++;
    }
}

static void processWeight(ParseState* s)
{
    std::string_view sv(s->charBuf);
    double* fe = s->fevt_arr + s->cur_event * (4 + s->n_weights);
    consume_next(sv, fe[4 + s->cur_weight++]);
    s->charBuf.clear();
    s->capture = NO_CAPTURE;
}

// -- SAX callbacks --

static void XMLCALL onStart(void* ud, const XML_Char* name, const XML_Char** attributes)
{
    ParseState* s = static_cast<ParseState*>(ud);

    if (std::strcmp(name, "event") == 0) {
        s->capture = EVENT_HEADER;
        s->cur_weight = 0;
    } else if (s->capture == EVENT_HEADER) {
        // for header, next <tag> is equivalent to onEnd(...) because header has no enclosing tag
        processEvent(s);
        // simularly, particle lines have no tag, so they must be processed without callbacks
        s->charBuf.clear(); //end of event-level data (remainder has enclosing tags)
        s->capture = NO_CAPTURE;
    }

    if (std::strcmp(name, "wgt") == 0)
        s->capture = WGT_TAG;
}

static void XMLCALL onEnd(void* ud, const XML_Char* name)
{
    ParseState* s = static_cast<ParseState*>(ud);

    if (std::strcmp(name, "event") == 0)
        s->cur_event++;
    else if (std::strcmp(name, "wgt") == 0)
        processWeight(s);
}

static void XMLCALL onChar(void* ud, const XML_Char* buf, int len)
{
    auto* s = static_cast<ParseState*>(ud);
    if (s->capture)
        s->charBuf.append(buf, len);
}

// ---------------------------------------------------------------------------
// Main entry point exposed to Python
// double passes LHE file, first to extract numbers of events, weights, and particles,
// then to read all values into preallocated arrays passed directly into nupmy structures by pybind11
// ---------------------------------------------------------------------------
py::tuple parseLHE(const std::string& filename)
{
    // --- Pass 1 ---
    auto [n_events, n_weights, n_particles] = countDimensions(filename);

    if (n_events == 0 || n_weights == 0 || n_particles == 0) 
        throw std::runtime_error("Found no events, weights, or particles.");

    // Double Arrays
    py::array_t<double> f_evt({n_events, 4 + n_weights});
    py::array_t<double> f_ptc({n_particles, 7});
    auto f_evt_buf = f_evt.request();
    auto f_ptc_buf = f_ptc.request();
    std::memset(f_evt_buf.ptr, 0, sizeof(double) * n_events * (4 + n_weights));
    std::memset(f_ptc_buf.ptr, 0, sizeof(double) * n_particles * 7);

    // Int Arrays
    py::array_t<int> i_evt({n_events, 2});
    py::array_t<int> i_ptc({n_particles, 7});
    auto i_evt_buf = i_evt.request();
    auto i_ptc_buf = i_ptc.request();
    std::memset(i_evt_buf.ptr, 0, sizeof(int) * n_events * 2);
    std::memset(i_ptc_buf.ptr, 0, sizeof(int) * n_particles * 7);

    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + filename);

    ParseState state;

    // Assigning to the state struct
    state.fevt_arr = static_cast<double*>(f_evt_buf.ptr);
    state.fptc_arr = static_cast<double*>(f_ptc_buf.ptr);
    state.ievt_arr    = static_cast<int*>(i_evt_buf.ptr);
    state.iptc_arr    = static_cast<int*>(i_ptc_buf.ptr);

    state.n_events  = n_events;
    state.n_weights = n_weights;
    state.n_particles = n_particles;

    XML_Parser parser = XML_ParserCreate(nullptr);
    XML_SetUserData(parser, &state);
    XML_SetElementHandler(parser, onStart, onEnd);
    XML_SetCharacterDataHandler(parser, onChar);

    constexpr size_t CHUNK = 65536;
    char chunk[CHUNK];
    bool parseError = false;
    std::string errorMsg;

    while (f.read(chunk, CHUNK) || f.gcount() > 0)
    {
        int bytes   = static_cast<int>(f.gcount());
        int isFinal = f.eof() ? 1 : 0;

        if (XML_Parse(parser, chunk, bytes, isFinal) == XML_STATUS_ERROR)
        {
            parseError = true;
            errorMsg   = std::string("Expat error at line ")
                       + std::to_string(XML_GetCurrentLineNumber(parser))
                       + ": "
                       + XML_ErrorString(XML_GetErrorCode(parser));
            break;
        }

        if (isFinal) break;
    }

    XML_ParserFree(parser);

    if (parseError)
        throw std::runtime_error(errorMsg);

    return py::make_tuple(i_evt, f_evt, i_ptc, f_ptc);
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(lhe_parser, m)
{
    m.doc() = "Fast LHE parser";
    m.def("parse_lhe", &parseLHE, py::arg("filename"),
          "Parse an LHE file.");
}