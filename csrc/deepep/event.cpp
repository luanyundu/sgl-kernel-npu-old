#include "event.hpp"

namespace deep_ep {

torch::Event create_event(const c10_npu::NPUStream& s) {
    auto event = torch::Event(c10::DeviceType::PrivateUse1);
    event.record(s);
    return event;
}

void stream_wait(const c10_npu::NPUStream& s_0, const c10_npu::NPUStream& s_1) {
    EP_HOST_ASSERT(s_0.id() != s_1.id());
    s_0.unwrap().wait(create_event(s_1));
}

void stream_wait(const c10_npu::NPUStream& s, const EventHandle& event) {
    s.unwrap().wait(*event.event);
}

}  // namespace deep_ep