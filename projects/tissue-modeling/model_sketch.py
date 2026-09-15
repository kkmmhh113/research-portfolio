"""TissuePK의 조직별 약물량 계산. 시간 h, 양 mg, 부피 L을 사용합니다."""


def equivalent_concentration(amount, volume, partition_coefficient):
    return max(amount, 0.0) / (volume * partition_coefficient)


def michaelis_menten(vmax, km, unbound_concentration):
    concentration = max(unbound_concentration, 0.0)
    return vmax * concentration / (km + concentration)


def intestinal_segment(
    lumen_amount, wall_amount, incoming_lumen, central_concentration,
    absorption_scale, absorption_rate, transit_rate,
    wall_volume, kp_gut, flow, fraction_unbound, vmax, km,
):
    """한 장 구간의 흡수, 다음 구간으로의 이동, 장벽 대사."""
    lumen = max(lumen_amount, 0.0)
    wall_concentration = equivalent_concentration(wall_amount, wall_volume, kp_gut)
    absorption = absorption_scale * absorption_rate * lumen
    transit = transit_rate * lumen
    venous_return = flow * wall_concentration
    metabolism = michaelis_menten(vmax, km, fraction_unbound * wall_concentration)

    lumen_change = incoming_lumen - absorption - transit
    wall_change = absorption + flow * central_concentration - venous_return - metabolism
    return lumen_change, wall_change, transit, venous_return, metabolism


def liver_zones(
    zone_amounts, liver_volume, kp_liver, central_concentration,
    portal_return, portal_flow, hepatic_artery_flow,
    fraction_unbound, total_vmax, km, zone_activity,
):
    """세 간 구간의 직렬 흐름. total_vmax는 생리 조건의 보정이 끝난 값."""
    flow = portal_flow + hepatic_artery_flow
    concentrations = [
        equivalent_concentration(amount, liver_volume / 3.0, kp_liver)
        for amount in zone_amounts
    ]
    activity_sum = sum(zone_activity)
    metabolism = [
        michaelis_menten(
            total_vmax * activity / activity_sum, km, fraction_unbound * concentration
        )
        for activity, concentration in zip(zone_activity, concentrations)
    ]
    c1, c2, c3 = concentrations
    v1, v2, v3 = metabolism
    changes = (
        portal_return + hepatic_artery_flow * central_concentration - flow * c1 - v1,
        flow * c1 - flow * c2 - v2,
        flow * c2 - flow * c3 - v3,
    )
    return changes, flow * c3, sum(metabolism)


def kidney_compartment(
    amount, volume, kp_kidney, central_concentration,
    renal_flow, apparent_clearance, renal_scale,
):
    """겉보기 청소율을 사용하는 단순화된 신장 구획."""
    concentration = equivalent_concentration(amount, volume, kp_kidney)
    elimination = apparent_clearance * renal_scale * concentration
    change = renal_flow * (central_concentration - concentration) - elimination
    return change, elimination
