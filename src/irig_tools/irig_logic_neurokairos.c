// Define feature test macros before any includes
#define _POSIX_C_SOURCE 199309L
#define _GNU_SOURCE
#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE

#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef enum {
    IRIG_ZERO = 0,
    IRIG_ONE = 1,
    IRIG_P = 2
} irig_bit_t;

typedef struct {
    double *data;
    size_t length;
    size_t capacity;
} double_array_t;

typedef struct {
    double utc_seconds;
    int pin_state;
    char local_datetime[96];
} transition_record_t;

typedef struct {
    transition_record_t *data;
    size_t length;
    size_t capacity;
} transition_array_t;

typedef struct {
    bool running;
    int simulated_pin_state;
    double_array_t encoded_times;
    double_array_t sending_starts;
    transition_array_t transitions;
    char encoded_timestamp_filename[256];
    char transition_log_filename[256];
    irig_bit_t current_frame[60];
    irig_bit_t next_frame[60];
    double pulse_lengths[60];
    uint64_t bit_start_times[60];
    int chrony_stratum;
    double chrony_root_dispersion;
    bool chrony_synced;
} irig_logic_neurokairos_t;

// Constants
#define SENDING_BIT_LENGTH 1.0
#define OFFSET_NS 20000
#define BUSY_WAIT_BUFFER_NS 1000000L
#define BUSY_WAIT_SLEEP_NS 0L

static const time_t DEFAULT_RUNTIME_SECONDS = 600;
static const time_t MAX_RUNTIME_SECONDS = 600;
static const uint64_t NS_PER_SEC = 1000000000ULL;
static uint64_t bit_length_ns;

static const int SECONDS_WEIGHTS[] = {1, 2, 4, 8, 10, 20, 40};
static const int MINUTES_WEIGHTS[] = {1, 2, 4, 8, 10, 20, 40};
static const int HOURS_WEIGHTS[] = {1, 2, 4, 8, 10, 20};
static const int DAY_OF_YEAR_WEIGHTS[] = {1, 2, 4, 8, 10, 20, 40, 80, 100, 200};
static const int DECISECONDS_WEIGHTS[] = {1, 2, 4, 8};
static const int YEARS_WEIGHTS[] = {1, 2, 4, 8, 10, 20, 40, 80};

volatile sig_atomic_t running = 1;
static int debug_mode = 0;

void init_timing_constants(void);
uint64_t timespec_to_ns(const struct timespec *ts);
void ultra_wait_until_ns(uint64_t target_ns);
void log_pin_transition(irig_logic_neurokairos_t *logic, int pin_state,
                        const struct timespec *transition_time);

void signal_handler(int sig) {
    printf("Received signal %d, shutting down gracefully...\n", sig);
    running = 0;
}

void init_double_array(double_array_t *arr, size_t initial_capacity) {
    arr->data = malloc(sizeof(double) * initial_capacity);
    arr->length = 0;
    arr->capacity = initial_capacity;
}

void append_double(double_array_t *arr, double value) {
    if (arr->length >= arr->capacity) {
        arr->capacity *= 2;
        arr->data = realloc(arr->data, sizeof(double) * arr->capacity);
    }
    arr->data[arr->length++] = value;
}

void init_transition_array(transition_array_t *arr, size_t initial_capacity) {
    arr->data = malloc(sizeof(transition_record_t) * initial_capacity);
    arr->length = 0;
    arr->capacity = initial_capacity;
}

void append_transition(transition_array_t *arr, double utc_seconds, int pin_state,
                       const char *local_datetime) {
    if (arr->length >= arr->capacity) {
        arr->capacity *= 2;
        arr->data = realloc(arr->data, sizeof(transition_record_t) * arr->capacity);
    }

    arr->data[arr->length].utc_seconds = utc_seconds;
    arr->data[arr->length].pin_state = pin_state;
    strncpy(arr->data[arr->length].local_datetime, local_datetime,
            sizeof(arr->data[arr->length].local_datetime) - 1);
    arr->data[arr->length].local_datetime[sizeof(arr->data[arr->length].local_datetime) - 1] = '\0';
    arr->length++;
}

void bcd_encode(int value, const int *weights, int weight_count, int *result) {
    memset(result, 0, weight_count * sizeof(int));
    for (int i = weight_count - 1; i >= 0; i--) {
        if (weights[i] <= value) {
            result[i] = 1;
            value -= weights[i];
        }
    }
}

int encode_stratum(int stratum) {
    if (stratum == 1) return 0;
    if (stratum == 2) return 1;
    if (stratum == 3) return 2;
    return 3;
}

int encode_root_dispersion(double dispersion_sec) {
    double dispersion_ms = dispersion_sec * 1000.0;
    if (dispersion_ms < 0.25) return 0;
    if (dispersion_ms < 0.5) return 1;
    if (dispersion_ms < 1.0) return 2;
    if (dispersion_ms < 2.0) return 3;
    if (dispersion_ms < 4.0) return 4;
    if (dispersion_ms < 8.0) return 5;
    if (dispersion_ms < 16.0) return 6;
    return 7;
}

void load_mock_chrony_status(irig_logic_neurokairos_t *logic) {
    logic->chrony_stratum = 1;
    logic->chrony_root_dispersion = 0.0001;
    logic->chrony_synced = true;
}

void generate_irig_h_frame(irig_logic_neurokairos_t *logic, const struct tm *time_info,
                           time_t frame_reference_second, irig_bit_t *frame) {
    int seconds_bcd[7], minutes_bcd[7], hours_bcd[6];
    int day_of_year_bcd[10], deciseconds_bcd[4], year_bcd[8];
    int pos = 0;
    int stratum_enc;
    int dispersion_enc;

    append_double(&logic->encoded_times, (double)frame_reference_second);

    bcd_encode(time_info->tm_sec, SECONDS_WEIGHTS, 7, seconds_bcd);
    bcd_encode(time_info->tm_min, MINUTES_WEIGHTS, 7, minutes_bcd);
    bcd_encode(time_info->tm_hour, HOURS_WEIGHTS, 6, hours_bcd);
    bcd_encode(time_info->tm_yday + 1, DAY_OF_YEAR_WEIGHTS, 10, day_of_year_bcd);
    bcd_encode(0, DECISECONDS_WEIGHTS, 4, deciseconds_bcd);
    bcd_encode((time_info->tm_year + 1900) % 100, YEARS_WEIGHTS, 8, year_bcd);

    frame[pos++] = IRIG_P;

    for (int i = 0; i < 4; i++) frame[pos++] = seconds_bcd[i] ? IRIG_ONE : IRIG_ZERO;
    frame[pos++] = IRIG_ZERO;
    for (int i = 4; i < 7; i++) frame[pos++] = seconds_bcd[i] ? IRIG_ONE : IRIG_ZERO;
    frame[pos++] = IRIG_P;

    for (int i = 0; i < 4; i++) frame[pos++] = minutes_bcd[i] ? IRIG_ONE : IRIG_ZERO;
    frame[pos++] = IRIG_ZERO;
    for (int i = 4; i < 7; i++) frame[pos++] = minutes_bcd[i] ? IRIG_ONE : IRIG_ZERO;
    frame[pos++] = IRIG_ZERO;
    frame[pos++] = IRIG_P;

    for (int i = 0; i < 4; i++) frame[pos++] = hours_bcd[i] ? IRIG_ONE : IRIG_ZERO;
    frame[pos++] = IRIG_ZERO;
    for (int i = 4; i < 6; i++) frame[pos++] = hours_bcd[i] ? IRIG_ONE : IRIG_ZERO;
    frame[pos++] = IRIG_ZERO;
    frame[pos++] = IRIG_ZERO;
    frame[pos++] = IRIG_P;

    for (int i = 0; i < 4; i++) frame[pos++] = day_of_year_bcd[i] ? IRIG_ONE : IRIG_ZERO;
    frame[pos++] = IRIG_ZERO;
    for (int i = 4; i < 8; i++) frame[pos++] = day_of_year_bcd[i] ? IRIG_ONE : IRIG_ZERO;
    frame[pos++] = IRIG_P;
    for (int i = 8; i < 10; i++) frame[pos++] = day_of_year_bcd[i] ? IRIG_ONE : IRIG_ZERO;

    frame[pos++] = IRIG_ZERO;
    frame[pos++] = IRIG_ZERO;
    frame[pos++] = IRIG_ZERO;
    for (int i = 0; i < 4; i++) frame[pos++] = deciseconds_bcd[i] ? IRIG_ONE : IRIG_ZERO;
    frame[pos++] = IRIG_P;

    for (int i = 0; i < 4; i++) frame[pos++] = year_bcd[i] ? IRIG_ONE : IRIG_ZERO;
    frame[pos++] = IRIG_ZERO;
    for (int i = 4; i < 8; i++) frame[pos++] = year_bcd[i] ? IRIG_ONE : IRIG_ZERO;
    frame[pos++] = IRIG_P;

    stratum_enc = encode_stratum(logic->chrony_stratum);
    frame[43] = (stratum_enc & 1) ? IRIG_ONE : IRIG_ZERO;
    frame[44] = (stratum_enc & 2) ? IRIG_ONE : IRIG_ZERO;

    dispersion_enc = encode_root_dispersion(logic->chrony_root_dispersion);
    frame[46] = (dispersion_enc & 1) ? IRIG_ONE : IRIG_ZERO;
    frame[47] = (dispersion_enc & 2) ? IRIG_ONE : IRIG_ZERO;
    frame[48] = (dispersion_enc & 4) ? IRIG_ONE : IRIG_ZERO;
}

void init_timing_constants(void) {
    bit_length_ns = (uint64_t)(SENDING_BIT_LENGTH * NS_PER_SEC);
}

uint64_t timespec_to_ns(const struct timespec *ts) {
    return (uint64_t)ts->tv_sec * NS_PER_SEC + (uint64_t)ts->tv_nsec;
}

void ultra_wait_until_ns(uint64_t target_ns) {
    struct timespec current_time;
    uint64_t current_ns;
    int64_t remaining_ns;
    struct timespec sleep_time;
    static const uint64_t MAX_SLEEP_NS = 100000000ULL;

    while (running) {
        clock_gettime(CLOCK_MONOTONIC, &current_time);
        current_ns = timespec_to_ns(&current_time);
        remaining_ns = (int64_t)(target_ns - current_ns);

        if (remaining_ns <= 0) break;

        if (remaining_ns > BUSY_WAIT_BUFFER_NS) {
            uint64_t sleep_duration = remaining_ns - BUSY_WAIT_BUFFER_NS;
            if (sleep_duration > MAX_SLEEP_NS) {
                sleep_duration = MAX_SLEEP_NS;
            }
            sleep_time.tv_sec = sleep_duration / NS_PER_SEC;
            sleep_time.tv_nsec = sleep_duration % NS_PER_SEC;
            nanosleep(&sleep_time, NULL);
        } else {
            break;
        }
    }

#if BUSY_WAIT_SLEEP_NS > 0
    sleep_time.tv_sec = 0;
    sleep_time.tv_nsec = BUSY_WAIT_SLEEP_NS;
    do {
        clock_gettime(CLOCK_MONOTONIC, &current_time);
        current_ns = timespec_to_ns(&current_time);
        if (current_ns < target_ns) {
            nanosleep(&sleep_time, NULL);
        }
    } while (current_ns < target_ns && running);
#else
    do {
        clock_gettime(CLOCK_MONOTONIC, &current_time);
        current_ns = timespec_to_ns(&current_time);
    } while (current_ns < target_ns && running);
#endif
}

double calculate_pulse_length(irig_bit_t bit) {
    switch (bit) {
        case IRIG_P: return 0.8 * SENDING_BIT_LENGTH;
        case IRIG_ONE: return 0.5 * SENDING_BIT_LENGTH;
        case IRIG_ZERO:
        default: return 0.2 * SENDING_BIT_LENGTH;
    }
}

void log_pin_transition(irig_logic_neurokairos_t *logic, int pin_state,
                        const struct timespec *transition_time) {
    struct tm local_tm;
    char datetime_prefix[48];
    char timezone_suffix[16];
    char local_datetime[96];
    double utc_seconds;

    if (localtime_r(&transition_time->tv_sec, &local_tm) == NULL) {
        return;
    }

    if (strftime(datetime_prefix, sizeof(datetime_prefix), "%Y-%m-%d %H:%M:%S", &local_tm) == 0) {
        return;
    }

    if (strftime(timezone_suffix, sizeof(timezone_suffix), "%Z", &local_tm) == 0) {
        timezone_suffix[0] = '\0';
    }

    snprintf(local_datetime, sizeof(local_datetime), "%s.%09ld %s",
             datetime_prefix, transition_time->tv_nsec, timezone_suffix);

    utc_seconds = (double)transition_time->tv_sec + (double)transition_time->tv_nsec * 1e-9;
    append_transition(&logic->transitions, utc_seconds, pin_state, local_datetime);
}

void simulate_pulse(irig_logic_neurokairos_t *logic, uint64_t pulse_duration_ns) {
    struct timespec start_time, current_time, sleep_time, transition_time;
    uint64_t start_ns, target_ns, current_ns;
    int64_t remaining_ns;
    static const uint64_t MAX_SLEEP_NS = 100000000ULL;

    clock_gettime(CLOCK_MONOTONIC, &start_time);
    start_ns = timespec_to_ns(&start_time);
    target_ns = start_ns + pulse_duration_ns;

    logic->simulated_pin_state = 1;
    clock_gettime(CLOCK_REALTIME, &transition_time);
    log_pin_transition(logic, logic->simulated_pin_state, &transition_time);

    while (running && logic->running) {
        clock_gettime(CLOCK_MONOTONIC, &current_time);
        current_ns = timespec_to_ns(&current_time);
        remaining_ns = (int64_t)(target_ns - current_ns);

        if (remaining_ns <= 0) break;

        if (remaining_ns > BUSY_WAIT_BUFFER_NS) {
            uint64_t sleep_duration = remaining_ns - BUSY_WAIT_BUFFER_NS;
            if (sleep_duration > MAX_SLEEP_NS) {
                sleep_duration = MAX_SLEEP_NS;
            }
            sleep_time.tv_sec = sleep_duration / NS_PER_SEC;
            sleep_time.tv_nsec = sleep_duration % NS_PER_SEC;
            nanosleep(&sleep_time, NULL);
        } else {
            break;
        }
    }

#if BUSY_WAIT_SLEEP_NS > 0
    {
        struct timespec poll_sleep;
        poll_sleep.tv_sec = 0;
        poll_sleep.tv_nsec = BUSY_WAIT_SLEEP_NS;

        do {
            clock_gettime(CLOCK_MONOTONIC, &current_time);
            current_ns = timespec_to_ns(&current_time);
            if (current_ns < target_ns) {
                nanosleep(&poll_sleep, NULL);
            }
        } while (current_ns < target_ns && running && logic->running);
    }
#else
    do {
        clock_gettime(CLOCK_MONOTONIC, &current_time);
        current_ns = timespec_to_ns(&current_time);
    } while (current_ns < target_ns && running && logic->running);
#endif

    logic->simulated_pin_state = 0;
    clock_gettime(CLOCK_REALTIME, &transition_time);
    log_pin_transition(logic, logic->simulated_pin_state, &transition_time);
}

void precalculate_next_frame(irig_logic_neurokairos_t *logic, time_t target_second) {
    struct tm time_info;
    char time_buffer[64];
    uint64_t frame_start_ns = (uint64_t)target_second * NS_PER_SEC;

    if (gmtime_r(&target_second, &time_info) == NULL) {
        return;
    }

    generate_irig_h_frame(logic, &time_info, target_second, logic->next_frame);

    if (debug_mode && asctime_r(&time_info, time_buffer) != NULL) {
        printf("encoding time (UTC): %s", time_buffer);
    }

    for (int i = 0; i < 60; i++) {
        logic->pulse_lengths[i] = calculate_pulse_length(logic->next_frame[i]);
        logic->bit_start_times[i] = frame_start_ns + (i * NS_PER_SEC) - OFFSET_NS;
    }
}

void write_logs_to_files(irig_logic_neurokairos_t *logic) {
    FILE *file = fopen(logic->encoded_timestamp_filename, "w");
    size_t min_length;

    if (!file) {
        printf("Could not open file for writing: %s\n", logic->encoded_timestamp_filename);
        return;
    }

    fprintf(file, "Encoded times,Sending starts\n");
    min_length = (logic->encoded_times.length < logic->sending_starts.length)
                 ? logic->encoded_times.length : logic->sending_starts.length;

    for (size_t i = 0; i < min_length; i++) {
        fprintf(file, "%f,%f\n", logic->encoded_times.data[i], logic->sending_starts.data[i]);
    }

    fclose(file);
    printf("Timestamps written to %s\n", logic->encoded_timestamp_filename);

    file = fopen(logic->transition_log_filename, "w");
    if (!file) {
        printf("Could not open file for writing: %s\n", logic->transition_log_filename);
        return;
    }

    fprintf(file, "utc_seconds,local_datetime,pin_state\n");
    for (size_t i = 0; i < logic->transitions.length; i++) {
        fprintf(file, "%.9f,%s,%d\n",
                logic->transitions.data[i].utc_seconds,
                logic->transitions.data[i].local_datetime,
                logic->transitions.data[i].pin_state);
    }

    fclose(file);
    printf("Timestamps written to %s\n", logic->transition_log_filename);
}

irig_logic_neurokairos_t *create_irig_logic_neurokairos(void) {
    irig_logic_neurokairos_t *logic = calloc(1, sizeof(irig_logic_neurokairos_t));
    time_t now;
    struct tm *tm_info;

    if (!logic) {
        return NULL;
    }

    logic->running = false;
    logic->simulated_pin_state = 0;
    load_mock_chrony_status(logic);

    init_double_array(&logic->encoded_times, 100);
    init_double_array(&logic->sending_starts, 100);
    init_transition_array(&logic->transitions, 1000);

    if (!logic->encoded_times.data || !logic->sending_starts.data || !logic->transitions.data) {
        free(logic->encoded_times.data);
        free(logic->sending_starts.data);
        free(logic->transitions.data);
        free(logic);
        return NULL;
    }

    now = time(NULL);
    tm_info = localtime(&now);
    strftime(logic->encoded_timestamp_filename, sizeof(logic->encoded_timestamp_filename),
             "irig_logic_neurokairos_output_timestamps_%Y-%m-%d_%H-%M-%S.csv", tm_info);
    strftime(logic->transition_log_filename, sizeof(logic->transition_log_filename),
             "irig_logic_neurokairos_transitions_%Y-%m-%d_%H-%M-%S.csv", tm_info);

    return logic;
}

void destroy_irig_logic_neurokairos(irig_logic_neurokairos_t *logic) {
    if (!logic) {
        return;
    }

    free(logic->encoded_times.data);
    free(logic->sending_starts.data);
    free(logic->transitions.data);
    free(logic);
}

int parse_runtime_seconds(const char *value, time_t *runtime_seconds) {
    char *end_ptr = NULL;
    long parsed_value = strtol(value, &end_ptr, 10);

    if (end_ptr == value || *end_ptr != '\0') {
        return -1;
    }

    if (parsed_value <= 0 || parsed_value > MAX_RUNTIME_SECONDS) {
        return -1;
    }

    *runtime_seconds = (time_t)parsed_value;
    return 0;
}

void print_usage(const char *program_name) {
    printf("Usage: %s [runtime_seconds] [debug]\n", program_name);
    printf("  runtime_seconds: optional integer from 1 to %ld (default %ld)\n",
           (long)MAX_RUNTIME_SECONDS, (long)DEFAULT_RUNTIME_SECONDS);
    printf("  debug: optional flag to print frame timing details\n");
}

int run_irig_simulation(irig_logic_neurokairos_t *logic, time_t max_runtime_seconds) {
    struct timespec current_time;
    struct timespec real_ts;
    struct timespec mono_ts;
    int64_t rt_to_mono_offset_ns;
    time_t next_frame_time;
    time_t stop_time;
    bool pulses_started = false;

    init_timing_constants();

    stop_time = time(NULL) + max_runtime_seconds;
    logic->running = true;

    printf("IRIG NeuroKairos logic simulator started\n");
    printf("Runtime limit: %ld seconds\n", (long)max_runtime_seconds);
    printf("Mock chrony status: stratum=%d dispersion=%.6f synced=%s\n",
           logic->chrony_stratum, logic->chrony_root_dispersion,
           logic->chrony_synced ? "yes" : "no");

    clock_gettime(CLOCK_REALTIME, &current_time);
    next_frame_time = ((current_time.tv_sec / 60) + 1) * 60;

    if (next_frame_time >= stop_time) {
        printf("Runtime ends before the next minute boundary; no pulses will be generated.\n");
        logic->running = false;
        return 0;
    }

    precalculate_next_frame(logic, next_frame_time);

    while (logic->running && running) {
        memcpy(logic->current_frame, logic->next_frame, sizeof(logic->current_frame));

        for (int i = 0; i < 60 && logic->running && running; i++) {
            uint64_t pulse_ns = (uint64_t)(logic->pulse_lengths[i] * NS_PER_SEC);
            uint64_t mono_target_ns;

            clock_gettime(CLOCK_REALTIME, &real_ts);
            clock_gettime(CLOCK_MONOTONIC, &mono_ts);
            rt_to_mono_offset_ns = (int64_t)timespec_to_ns(&mono_ts)
                                 - (int64_t)timespec_to_ns(&real_ts);
            mono_target_ns = logic->bit_start_times[i] + rt_to_mono_offset_ns;

            if ((time_t)(logic->bit_start_times[i] / NS_PER_SEC) >= stop_time) {
                logic->running = false;
                break;
            }

            ultra_wait_until_ns(mono_target_ns);

            if (!logic->running || !running) {
                break;
            }

            if (i == 0) {
                struct timespec start_ts;
                clock_gettime(CLOCK_REALTIME, &start_ts);
                append_double(&logic->sending_starts,
                              (double)start_ts.tv_sec + (double)start_ts.tv_nsec * 1e-9);
            }

            if (debug_mode) {
                struct timespec debug_time;
                clock_gettime(CLOCK_REALTIME, &debug_time);
                printf("Bit %d simulated at system time: %ld.%09ld, state -> 1 then 0\n",
                       i, debug_time.tv_sec, debug_time.tv_nsec);
            }

            simulate_pulse(logic, pulse_ns);
            pulses_started = true;
        }

        next_frame_time += 60;
        if (next_frame_time >= stop_time) {
            break;
        }
        precalculate_next_frame(logic, next_frame_time);
    }

    logic->running = false;
    if (!pulses_started) {
        printf("IRIG NeuroKairos logic simulator stopped before first pulse\n");
    } else {
        printf("IRIG NeuroKairos logic simulator stopping\n");
    }
    return 0;
}

int main(int argc, char *argv[]) {
    irig_logic_neurokairos_t *logic;
    time_t runtime_seconds = DEFAULT_RUNTIME_SECONDS;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "debug") == 0) {
            debug_mode = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (parse_runtime_seconds(argv[i], &runtime_seconds) != 0) {
            print_usage(argv[0]);
            return 1;
        }
    }

    signal(SIGTERM, signal_handler);
    signal(SIGINT, signal_handler);

    logic = create_irig_logic_neurokairos();
    if (!logic) {
        printf("Failed to initialize IRIG NeuroKairos logic simulator\n");
        return 1;
    }

    run_irig_simulation(logic, runtime_seconds);
    write_logs_to_files(logic);
    destroy_irig_logic_neurokairos(logic);

    return 0;
}
